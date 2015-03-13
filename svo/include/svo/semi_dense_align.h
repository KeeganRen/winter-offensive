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

#ifndef SVO_SEMI_DENSE_ALIGN_H_
#define SVO_SEMI_DENSE_ALIGN_H_

#include <vikit/nlls_solver.h>
#include <vikit/performance_monitor.h>
#include <svo/global.h>

namespace vk {
class AbstractCamera;
}

namespace svo {

class Feature;
struct Seed;

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SemiDenseAlign : public vk::NLLSSolver<6, SE3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  cv::Mat resimg_;

  struct Options{
      bool weighted;
      bool robust;
      double intensity_err_squared;
      double dep_var_scale;

      Options():
          weighted(false),
          robust(false),
          intensity_err_squared(0),
          dep_var_scale(10000.0)
      {}
  }options_;

  SemiDenseAlign(
      int n_levels,
      int min_level,
      int n_iter,
      Method method,
      bool display,
      bool verbose);

  size_t run(
      FramePtr ref_frame,
      FramePtr cur_frame);

  /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
  /// at the converged state.
  Matrix<double, 6, 6> getFisherInformation();

protected:
  FramePtr ref_frame_;            //!< reference frame, has depth for gradient pixels.
  FramePtr cur_frame_;            //!< only the image is known!
  int level_;                     //!< current pyramid level on which the optimization runs.
  bool display_;                  //!< display residual image.
  int max_level_;                 //!< coarsest pyramid level for the alignment.
  int min_level_;                 //!< finest pyramid level for the alignment.

  // cache:
  Matrix<double, 6, Dynamic, ColMajor> jacobian_cache_;
  bool have_ref_patch_cache_;
  cv::Mat ref_patch_cache_;
  std::vector<bool> visible_fts_;
  std::vector<double> weight_cache_;

  // YS: compute ref_patch_cache_ and jacobian_cache_
  void precomputeReferencePatches(); 
  virtual double computeResiduals(const SE3& model, bool linearize_system, bool compute_weight_scale = false);
  virtual int solve();
  virtual void update (const ModelType& old_model, ModelType& new_model);
  virtual void startIteration();
  virtual void finishIteration();

//    ofstream log_file;
};

} // namespace svo

#endif // SVO_SEMI_DENSE_ALIGN_H_
