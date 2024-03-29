// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/feature_detection.h>
#include <svo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>

namespace svo {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

// YS: http://stackoverflow.com/questions/10250519/invoke-stdfunction-in-an-stdfor-each
// YS: http://msdn.microsoft.com/zh-cn/library/e5sk9w9k.aspx
// YS: http://msdn.microsoft.com/zh-cn/library/dd293608.aspx
void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    Features& fts)
{
    // YS: corners contains grid_n_cols_*grid_n_rows_ elements of default value
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    vector<fast::fast_xy> fast_corners;
#if __SSE2__
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 40, fast_corners);
#elif HAVE_FAST_NEON
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif
    vector<int> scores, nm_corners;
    fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 40, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k])
        continue;
      const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }

  // Create feature for every corner that has high enough corner score
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    if(c.score > detection_threshold)
      fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
  });

  resetGrid();
}

EdgeDetector::EdgeDetector(
        const int img_width,
        const int img_height,
        const int cell_size,
        const int n_pyr_levels) :
            AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{
    grad_thresh_ = 450;
}

void EdgeDetector::detect(
        Frame* frame,
        const ImgPyr& img_pyr,
        const double detection_threshold,
        Features& fts)
{
    Edges edges;
    for (int L=0; L<n_pyr_levels_; ++L)
    {
        const int scale = (1<<L);
        cv::Mat gradx, grady;

        cv::Sobel(img_pyr[L], gradx, CV_16S, 1, 0, 3);
        cv::Sobel(img_pyr[L], grady, CV_16S, 0, 1, 3);

        cv::Size size = img_pyr[L].size();

        for (int i=0; i<size.height; i++)
        {
            const int16_t* dxdata = (const int16_t*)(gradx.data + i*gradx.step);
            const int16_t* dydata = (const int16_t*)(grady.data + i*grady.step);

            for (int j=0; j<size.width; j++)
            {
                int16_t gx = dxdata[j];
                int16_t gy = dydata[j];

                int grad_norm_squared = gx*gx + gy*gy;
                if (grad_norm_squared > grad_thresh_*grad_thresh_)
                {
                    Edge edge(j*scale, i*scale,
                            sqrt(grad_norm_squared), L, Vector2d(gx,gy));
                    int key = edge.x + edge.y * img_pyr[0].cols;
                    auto occupancy = edges.find(key);
                    if(occupancy != edges.end())
                    {
                        if (occupancy->second.score < edge.score)
                        {
                             occupancy->second.grad_ = edge.grad_;
                             occupancy->second.score = edge.score;
                             occupancy->second.level = edge.level;
                        }
                    }
                    else
                        edges.insert(std::make_pair(key, edge));
                }
            }
        }
    }

    for (auto it=edges.begin(), ite=edges.end(); it != ite; ++it)
    {
        Feature* tmp_fts = new Feature(frame, Vector2d(it->second.x, it->second.y), it->second.level);
        tmp_fts->grad = it->second.grad_;
        tmp_fts->grad_mag = it->second.score;
        tmp_fts->type = Feature::EDGELET;
        fts.push_back(tmp_fts);
    }

    if (fts.size() > 4000)
        grad_thresh_ *= 1.05;
    else if(fts.size() < 2500)
        grad_thresh_ *= 0.95;
}

} // namespace feature_detection
} // namespace svo

