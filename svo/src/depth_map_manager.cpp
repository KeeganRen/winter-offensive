#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/depth_map_manager.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>
#include <svo/map.h>

namespace svo {
    DepthMapManager::DepthMapManager(feature_detection::DetectorPtr edge_detector) :
        edge_detector_(edge_detector),
        thread_(NULL),
        depth_map_updating_halt_(false),
        new_keyframe_set_(false),
        new_keyframe_mean_depth_(0.0),
        new_keyframe_min_depth_(0.0)
    {
#ifdef SVO_TRACE
        permon_.addTimer("edge_detection");
        permon_.addTimer("depth_map_update");
        permon_.addLog("initialized_n_edge");
        permon_.addLog("prior_num");
        permon_.addLog("update_n_edge");
        permon_.addLog("baseline_width");
        permon_.addLog("act_depth_mean");
        permon_.init("depth_map", "/tmp");
#endif
    }

    DepthMapManager::~DepthMapManager()
    {
        stopThread();
        SVO_INFO_STREAM("DepthMapManager destructed.");
    }

    void DepthMapManager::startThread()
    {
        thread_ = new boost::thread(&DepthMapManager::updateDMapLoop, this);
    }

    void DepthMapManager::stopThread()
    {
        SVO_INFO_STREAM("DepthManager stop thread invoked.");
        if (thread_ != NULL)
        {
            SVO_INFO_STREAM("DepthMapManager interrupt and join thread...");
            depth_map_updating_halt_ = true;
            thread_->interrupt();
            thread_->join();
            thread_ = NULL;
        }
    }

    void DepthMapManager::addFrame(FramePtr frame, double depth_mean, double depth_min)
    {
        FrameCandidate frame_candi={frame, depth_mean, depth_min};

        if (thread_ != NULL)
        {
            {
                lock_t lock(frame_queue_mut_);
                if (frame_queue_.size() > 2)
                {
                    SVO_WARN_STREAM("drop frame!");
                    frame_queue_.pop();
                }
                frame_queue_.push(frame_candi);
            }
            frame_queue_cond_.notify_one();
        }
    }

    void DepthMapManager::setActiveFrame(FramePtr frame, double depth_mean, double depth_min)
    {
        if (frame == active_keyframe_)
        {
            SVO_INFO_STREAM("same keyframe, ignored.");
            return;
        }
        new_keyframe_min_depth_ = depth_min;
        new_keyframe_mean_depth_ = depth_mean;
        if(thread_ != NULL)
        {
            new_keyframe_ = frame;
            new_keyframe_set_ = true;
            frame_queue_cond_.notify_one();
        }
    }

    void DepthMapManager::updateDMapLoop()
    {
        while (!boost::this_thread::interruption_requested())
        {
            FrameCandidate frame_candi;
            {
                lock_t lock(frame_queue_mut_);
                while ((frame_queue_.empty() && new_keyframe_set_==false) || depth_map_updating_halt_)  
                    frame_queue_cond_.wait(lock);
                if(new_keyframe_set_)
                {
                    new_keyframe_set_ = false;
                    clearFrameQueue();
                    initializeDepthMap(new_keyframe_, new_keyframe_mean_depth_, new_keyframe_min_depth_);
                    continue;
                }
                else
                {
                    frame_candi = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            updateMap(frame_candi.frame);
        }
    }

    void DepthMapManager::pauseUpdate()
    {
        depth_map_updating_halt_ = true;
    }

    void DepthMapManager::resumeUpdate()
    {
        depth_map_updating_halt_ = false;
        frame_queue_cond_.notify_one();
    }

    void DepthMapManager::reset()
    {
        depth_map_updating_halt_ = true;
        {
            lock_t lock(frame_queue_mut_);
            while(!frame_queue_.empty())
                frame_queue_.pop();
        }
        depth_map_updating_halt_ = false;

        if (options_.verbose)
            SVO_INFO_STREAM("DepthMapManager: RESET.");
    }

    // new keyframe's depth map initializing
    // prior information has gained in updateDepthMap
    void DepthMapManager::initializeDepthMap(FramePtr frame, double depth_mean, double depth_min)
    {
        if (!frame)
        {
            SVO_WARN_STREAM("Are you kidding me??");
            return;
        }
        if (!active_keyframe_)  // not started
        {   
            if (options_.verbose)
                SVO_INFO_STREAM("first depth map init");
            //extract pixels with salient gradient
            Features new_features;
#ifdef SVO_TRACE
            permon_.startTimer("edge_detection");
#endif
            edge_detector_->detect(frame.get(), frame->img_pyr_,
                    0.0, new_features);
#ifdef SVO_TRACE
            permon_.stopTimer("edge_detection");
            permon_.log("initialized_n_edge", new_features.size());
            permon_.log("prior_num", 0);
#endif
            depth_map_updating_halt_ = true;
            {
                lock_t lock(frame->depth_map_mut_);
                // new keyframe without prior info
                // in case adding the first keyframe
                // this will not happend in normal cycle
                std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
                        frame->depth_map_.push_back(Seed(ftr, depth_mean, depth_min));
                        });
            }
            active_keyframe_ = frame;
            depth_map_updating_halt_ = false;
            if (options_.verbose)
                SVO_INFO_STREAM("depth map random init OK.");
        }
        else
        {
            if (!frame->depth_map_.empty())
            {
                if (options_.verbose)
                    SVO_INFO_STREAM("previous map found.");
                active_keyframe_ = frame;
                return;
            }
            //extract pixels with salient gradient
            Features new_features;
#ifdef SVO_TRACE
            permon_.startTimer("edge_detection");
#endif
            edge_detector_->detect(frame.get(), frame->img_pyr_,
                    0.0, new_features);
#ifdef SVO_TRACE
            permon_.stopTimer("edge_detection");
            permon_.log("initialized_n_edge", new_features.size());
#endif
            // prepare to propagate
            depth_map_updating_halt_ = true;
            cv::Mat temp_depth_map=cv::Mat(480, 640, CV_64F, double(0));
            cv::Mat temp_variance_map=cv::Mat(480, 640, CV_64F, double(0));

            SE3 T_cur_ref = frame->T_f_w_ * active_keyframe_->T_f_w_.inverse();

            {
                lock_t lock_(active_keyframe_->depth_map_mut_);
                // new keyframe without prior info
                // in case adding the first keyframe
                // this will not happend in normal cycle
                for (auto it=active_keyframe_->depth_map_.begin(), ite=active_keyframe_->depth_map_.end(); it != ite; ++it)
                {
                    if (!it->converged)
                        continue;

                    Vector3d new_pt = T_cur_ref * (it->ftr->f/it->mu);
                    Vector2d new_px(frame->cam_->world2cam(vk::project2d(new_pt)));
                    const int u_cur_i = floorf(new_px[0]+0.5);
                    const int v_cur_i = floorf(new_px[1]+0.5);

                    // ignore those out of bound
                    if (u_cur_i-1<0 || v_cur_i-1<0 || u_cur_i+1>=active_keyframe_->cam_->width() || v_cur_i+1>=active_keyframe_->cam_->height())
                    {
                        continue;
                    }

                    // variance propagate
                    double d1_d0 = 1.0/(new_pt[2] * it->mu);
                    double d1_d0_2 = d1_d0*d1_d0;
                    double d1_d0_4 = d1_d0_2 * d1_d0_2;
                    double new_sigma2 = d1_d0_4 * it->sigma2 + 0.008;
                    temp_depth_map.at<double>(v_cur_i, u_cur_i) = new_pt[2];
                    temp_variance_map.at<double>(v_cur_i, u_cur_i) = new_sigma2;
                }
            }
            if (options_.verbose)
            {
                SVO_INFO_STREAM("propagated to Mat.");
                SVO_INFO_STREAM("new features number: "<<new_features.size());
            }
            {
                int prior_num=0;
                lock_t lock(frame->depth_map_mut_);
                for (auto it=new_features.begin(), ite=new_features.end(); it != ite; )
                {
                    if ((*it)->px[0]-1<0 || (*it)->px[1]-1<0 || (*it)->px[0]+1>=frame->cam_->width() || (*it)->px[1]+1>=frame->cam_->height())
                    {
                        delete *it;
                        it = new_features.erase(it);
                        continue;
                    }

                    int n_prior = 0;
                    double depth_sum = 0.0;
                    double depth_weight = 0.0;
                    double weight_sum = 0.0;
                    double min_variance = std::numeric_limits<double>::max();

//                        SVO_INFO_STREAM("debug point 1");
                    for (int i=0; i<3; i++)
                        for (int j=0; j<3; j++)
                        {
//                                SVO_INFO_STREAM((*it)->px[1]-1+i<<' '<<(*it)->px[0]-1+j);
                            const double d=temp_depth_map.at<double>((*it)->px[1]-1+i, (*it)->px[0]-1+j);
                            const double v=temp_variance_map.at<double>((*it)->px[1]-1+i,(*it)->px[0]-1+j);

                            if (d > 0.00001 && v>0.008) // sanity check
                            {
                                depth_weight = 1/v;
                                depth_sum += depth_weight*d;
                                if (min_variance > v)
                                    min_variance = v;
                                weight_sum += depth_weight;
                                n_prior ++;
                            }
                        }

//                       SVO_INFO_STREAM("debug point 2");
                    if (n_prior > 0)
                    {
                        prior_num++;
                        frame->depth_map_.push_back(Seed(*it, depth_sum/weight_sum, 1/(6*sqrt(min_variance))));
                    }
                    else
                    {
                        frame->depth_map_.push_back(Seed(*it, depth_mean, depth_min));
                    }
                     ++it;
                }
#ifdef SVO_TRACE
                permon_.log("prior_num", prior_num);
#endif
            }
            depth_map_updating_halt_ = false;
            active_keyframe_ = frame;
            if (options_.verbose)
                SVO_INFO_STREAM("depth map propagate success.");
        }
    }

    void DepthMapManager::updateMap(FramePtr frame)
    {
#ifdef SVO_TRACE
        int n_updated = 0;
        permon_.startTimer("depth_map_update");
#endif
        if (!active_keyframe_)
        {
            if (options_.verbose)
                SVO_INFO_STREAM("no proper keyframe.");
#ifdef SVO_TRACE
            permon_.stopTimer("depth_map_update");
            permon_.log("update_n_edge", -3);
            permon_.writeToFile();
#endif
            return;
        }
        
        double baseline_width = (frame->T_f_w_.translation() - active_keyframe_->T_f_w_.translation()).norm();
        double act_depth_mean, act_depth_min;
        getFrameDepth(active_keyframe_, act_depth_mean, act_depth_min);
        if (baseline_width/act_depth_mean > Config::minBaselineToDepthRatio())
        {
#ifdef SVO_TRACE
            permon_.stopTimer("depth_map_update");
            permon_.log("update_n_edge", -2);
            permon_.log("baseline_width", baseline_width);
            permon_.log("act_depth_mean",act_depth_mean);
            permon_.writeToFile();
#endif
            return;
        }

        lock_t lock(active_keyframe_->depth_map_mut_);  //TODO: guarantee read/write protection
        auto it = active_keyframe_->depth_map_.begin();

        const double focal_length = frame->cam_->errorMultiplier2();
        double px_noise = 1.5;
        double px_error_angle = atan(px_noise/(2.0*focal_length)) * 2.0;

        while (it != active_keyframe_->depth_map_.end())
        {
            if (depth_map_updating_halt_)
            {
#ifdef SVO_TRACE
                permon_.stopTimer("depth_map_update");
                permon_.log("update_n_edge", n_updated);
                permon_.writeToFile();
#endif
                return;
            }
            SE3 T_ref_cur = active_keyframe_->T_f_w_ * frame->T_f_w_.inverse();
            const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f)); // xyz in cur frame
            if (xyz_f.z() < 0.0)
            {
                ++it;   // behind the camera
                continue;
            }
            if (!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>(), 10))
            {
                ++it;
                continue;
            }

            float z_inv_min = it->mu + sqrt(it->sigma2);
            float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
            double z;
            Vector2d px_found;

            if (isnan(z_inv_min))
            {
                SVO_WARN_STREAM("z_min is NaN");
                delete it->ftr;
                it = active_keyframe_->depth_map_.erase(it);
                continue;
            }

            Matcher::MatchResult mres = matcher_.findEpipolarMatchDirect(*active_keyframe_, *frame
                    , *it->ftr
                    , 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, px_found, z);

            if (mres == Matcher::EdgeDirectionViolate)  // we do not penalize those gradient not match
            {
                ++it;
                continue;
            }
            else if (mres != Matcher::Success)
            {
                it->b++;
                if (it->a/(it->a+it->b) < 0.1) // inlier ratio too low
                {
                    delete it->ftr;
                    it = active_keyframe_->depth_map_.erase(it);
                }
                else
                    ++it;
                continue;
            }

            double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
            double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

            // update the estimate
            updateSeed(1./z, tau_inverse*tau_inverse, &(*it));

            if(sqrt(it->sigma2) < Config::edgeInverseDepthVarAccept()*it->z_range && it->converged == false)
            {
                active_keyframe_->depth_map_quality_ ++;
                it->converged = true;
            }

#ifdef SVO_TRACE
            n_updated ++;
#endif
            ++it;
        }
#ifdef SVO_TRACE
        permon_.stopTimer("depth_map_update");
        permon_.log("update_n_edge", n_updated);
        permon_.writeToFile();
#endif
    }

    void DepthMapManager::getFrameDepth(FramePtr frame, double& depth_mean, double& depth_min)
    {
        int good_depth_num=0;
        double depth_sum=0.0;

        depth_min = std::numeric_limits<double>::max();

        {
            lock_t lock(frame->depth_map_mut_);
            for(auto it=frame->depth_map_.begin(), ite=frame->depth_map_.end(); it!=ite; ++it)
            {
                if (it->converged)
                {
                    good_depth_num++;
                    const double z=1./it->mu;
                    depth_sum += z;
                    depth_min = fmin(z, depth_min);
                }
            }
        }
        depth_mean = depth_sum/good_depth_num;
    }

    void DepthMapManager::clearFrameQueue()
    {
        while (!frame_queue_.empty())
            frame_queue_.pop();
    }
}
