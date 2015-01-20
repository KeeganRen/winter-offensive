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
        permon_.addLog("min_sigma2");
        permon_.addLog("updated_depth_mean");
        permon_.addLog("initialize_depth_mean");
        permon_.addLog("prior_depth_mean");
        permon_.addLog("baseline_width");
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

    void DepthMapManager::addFrame(FramePtr frame)
    {
        if (thread_ != NULL)
        {
            {
                lock_t lock(frame_queue_mut_);
                if (frame_queue_.size() > 2)
                {
                    SVO_WARN_STREAM("drop frame!");
                    frame_queue_.pop();
                }
                frame_queue_.push(frame);
            }
            depth_map_updating_halt_ = false;
            frame_queue_cond_.notify_one();
        }
        else
            updateMap(frame);
    }

    void DepthMapManager::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
    {
        new_keyframe_mean_depth_ = depth_mean;
        new_keyframe_min_depth_ = depth_min;

        if (thread_ != NULL)
        {
            new_keyframe_ = frame;
            new_keyframe_set_ = true;
            depth_map_updating_halt_ = true;
            frame_queue_cond_.notify_one();
        }
        else
            initializeDepthMap(frame);
    }

    void DepthMapManager::removeKeyframe(FramePtr frame)
    {
        if (frame == active_keyframe_) //rarely happen
        {
            depth_map_updating_halt_ = true;
            active_keyframe_ = FramePtr();
        }
    }

    void DepthMapManager::updateDMapLoop()
    {
        while (!boost::this_thread::interruption_requested())
        {
            FramePtr frame;
            {
                lock_t lock(frame_queue_mut_);
                while (frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);
                if (new_keyframe_set_)
                {
                    new_keyframe_set_ = false;
                    depth_map_updating_halt_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            updateMap(frame);
            if (frame->isKeyframe())
                initializeDepthMap(frame);
        }
    }

    void DepthMapManager::reset()
    {
        depth_map_updating_halt_ = true;
        {
            lock_t lock(keyframe_neighbour_mut_);
            keyframe_neighbour_.clear();
        }

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
    void DepthMapManager::initializeDepthMap(FramePtr frame)
    {
        if (!frame->edge_extracted_)
        {   
            //extract pixels with salient gradient
            Features new_features;
#ifdef SVO_TRACE
            permon_.startTimer("edge_detection");
#endif
            edge_detector_->detect(frame.get(), frame->img_pyr_,
                    0.0, new_features);
            frame->edge_extracted_ = true;
#ifdef SVO_TRACE
            permon_.stopTimer("edge_detection");
#endif
            depth_map_updating_halt_ = true;
            {
                lock_t lock(frame->depth_map_mut_);
                // new keyframe without prior info
                // in case adding the first keyframe
                // this will not happend in normal cycle
                if (frame->depth_map_.empty())
                {
                    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
                            frame->depth_map_.insert(
                                make_pair(
                                    ftr->px[0]+ftr->px[1]*frame->cam_->width(), 
                                    new Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_)));
                            });
                }
                else
                {
#ifdef SVO_TRACE
                    permon_.log("prior_num", frame->depth_map_.size());
#endif
                    // make a seed for each feature
                    for(auto it=new_features.begin(), ite=new_features.end(); it != ite; ++it)
                    {
                        int key = (*it)->px[0] + (*it)->px[1]*frame->cam_->width();
                        int key_around[8];
                        key_around[0] = key-1-frame->cam_->width();
                        key_around[1] = key-frame->cam_->width();
                        key_around[2] = key+1-frame->cam_->width();
                        key_around[3] = key-1;
                        key_around[4] = key+1;
                        key_around[5] = key-1+frame->cam_->width();
                        key_around[6] = key+frame->cam_->width();
                        key_around[7] = key+1+frame->cam_->width();
                        Seed* new_seed_ptr;

                        int n_obs=0; // the number of neighbour pixels which has a prior depth estimation
                        double depth_weighted_sum = 0.0;
                        double depth_var_sum = 0.0;
                        double weight_sum = 0.0;

                        // if there is an estimation already in the same pixel location
                        // we don't need to allocate a new seed in memory
                        auto prior_seed_itr = frame->depth_map_.find(key);
                        if (prior_seed_itr != frame->depth_map_.end())
                        {
                            new_seed_ptr = prior_seed_itr->second;
                            new_seed_ptr->batch_id = 0; // mark the seed as reserved, those has batch_id=-1 will be deleted later

                            delete new_seed_ptr->ftr;   // release old feature memory
                            new_seed_ptr->ftr = *it;    // assign the newly detected feature
                                                        // TODO maybe a smart pointer would be better?
                            depth_weighted_sum = 1.0/(new_seed_ptr->sigma2 * new_seed_ptr->mu);
                            weight_sum = 1.0/new_seed_ptr->sigma2;
                            depth_var_sum = new_seed_ptr->sigma2;
                            n_obs = 1;
                        }
                        else
                        {
                            new_seed_ptr = new Seed(*it, new_keyframe_mean_depth_, new_keyframe_min_depth_);
                            frame->depth_map_.insert(make_pair(key, new_seed_ptr));
                        }

                        for( int i=0; i<8; i++)
                        {
                            auto prior_info = frame->depth_map_.find(key_around[i]);
                            if (prior_info != frame->depth_map_.end())
                            {
                                depth_weighted_sum += 1.0/(prior_info->second->sigma2 * prior_info->second->mu);
                                weight_sum += 1.0/prior_info->second->sigma2;
                                depth_var_sum += prior_info->second->sigma2;
                                n_obs++;
                            }
                        }
                        // has neighbour prior info
                        if (n_obs > 0)
                        {
                            new_seed_ptr->mu = weight_sum / depth_weighted_sum;
                            new_seed_ptr->sigma2 = depth_var_sum / n_obs;
                        }
                    }
                    // delete seed with batch_id = -1
                    for (auto it=frame->depth_map_.begin(), ite=frame->depth_map_.end(); it != ite; )
                    {
                        if (it->second->batch_id == -1)
                        {
                            delete it->second->ftr;
                            delete it->second;
                            it = frame->depth_map_.erase(it);
                        }
                        else
                            ++it;
                    }
                }
            }

#ifdef SVO_TRACE
            permon_.log("initialized_n_edge", frame->depth_map_.size());
            permon_.log("initialize_depth_mean", new_keyframe_mean_depth_);
#endif
        }
        else    // exits in neighbour
        {
            lock_t lock(keyframe_neighbour_mut_);
            for(auto it=keyframe_neighbour_.begin(), ite=keyframe_neighbour_.end(); it != ite; ++it)
            {
                if (*it == frame)
                    return;
            }
        }
        lock_t lock(keyframe_neighbour_mut_);
        if (keyframe_neighbour_.size() > 3)
            keyframe_neighbour_.pop_front();
        keyframe_neighbour_.push_back(frame);
    }

    void DepthMapManager::updateMap(FramePtr frame)
    {
        double baseline_width=-1.0; 
#ifdef SVO_TRACE
        int n_updated = 0;
        permon_.startTimer("depth_map_update");
#endif
        lock_t lock(keyframe_neighbour_mut_);
        for (auto kf_it=keyframe_neighbour_.begin(), kf_ite=keyframe_neighbour_.end(); kf_it!=kf_ite; ++kf_it)
        {           
            active_keyframe_ = *kf_it;
            if (!active_keyframe_)
            {
                if (options_.verbose)
                    SVO_INFO_STREAM("no proper keyframe.");
                return;
            }
            
            lock_t lock(active_keyframe_->depth_map_mut_);  //TODO: guarantee read/write protection
            auto it = active_keyframe_->depth_map_.begin();

            // small baseline width results larger uncertainty
            baseline_width = (active_keyframe_->T_f_w_.translation() - frame->T_f_w_.translation()).norm();
            if (baseline_width < Config::minBaselineToDepthRatio()*new_keyframe_mean_depth_)
                continue;
#ifdef SVO_TRACE
        permon_.log("baseline_width", baseline_width);
#endif
            const double focal_length = frame->cam_->errorMultiplier2();
            double px_noise = 1.5;
            double px_error_angle = atan(px_noise/(2.0*focal_length)) * 2.0;
            int good_edge=0;

            while (it != active_keyframe_->depth_map_.end())
            {
                if (depth_map_updating_halt_)
                    return;
                SE3 T_ref_cur = active_keyframe_->T_f_w_ * frame->T_f_w_.inverse();
                const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->second->mu * it->second->ftr->f)); // xyz in cur frame
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

                float z_inv_min = it->second->mu + sqrt(it->second->sigma2);
                float z_inv_max = max(it->second->mu - sqrt(it->second->sigma2), 0.00000001f);
                double z;
                Vector2d px_found;

                if (isnan(z_inv_min))
                {
                    SVO_WARN_STREAM("z_min is NaN");
                    delete it->second->ftr;
                    delete it->second;
                    it = active_keyframe_->depth_map_.erase(it);
                    continue;
                }

                Matcher::MatchResult mres = matcher_.findEpipolarMatchDirect(*active_keyframe_, *frame, *it->second->ftr,
                        1.0/it->second->mu, 1.0/z_inv_min, 1.0/z_inv_max, px_found, z);
                
                if (mres == Matcher::EdgeDirectionViolate)  // we do not penalize those gradient not match
                {
                    ++it;
                    continue;
                }
                else if (mres != Matcher::Success)
                {
                    it->second->b++;
                    if (it->second->a/(it->second->a+it->second->b) < 0.1) // inlier ratio too low
                    {
                        delete it->second->ftr;
                        delete it->second;
                        it = active_keyframe_->depth_map_.erase(it);
                    }
                    else
                        ++it;
                    continue;
                }
                
                double tau = computeTau(T_ref_cur, it->second->ftr->f, z, px_error_angle);
                double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

                // update the estimate
                updateSeed(1./z, tau_inverse*tau_inverse, it->second);

                if(sqrt(it->second->sigma2) < 0.005*it->second->z_range)
                {
                    it->second->converged = true;
                    good_edge++;
                }

#ifdef SVO_TRACE
                n_updated ++;
#endif
                // add the depth map pixel to current frame
                if (frame->isKeyframe())
                {
                    lock_t lock(frame->depth_map_mut_);

                    Vector2i px_ipos = (px_found + Vector2d(0.5, 0.5)).cast<int>();
                    int key = px_ipos[0] + px_ipos[1]*frame->cam_->width();
                    double new_mu = 1.0/(T_ref_cur.inverse()*(1.0/it->second->mu * it->second->ftr->f))[2];
                    double d1_d0 = new_mu / it->second->mu;
                    double d1_d0_2 = d1_d0*d1_d0;
                    double d1_d0_4 = d1_d0_2 * d1_d0_2; 
                    double new_sigma2 = d1_d0_4*it->second->sigma2 + 0.008;

                    auto exist_element = frame->depth_map_.find(key);
                    if (exist_element != frame->depth_map_.end())   //pixel location occupied, update the estimation
                    {
                        exist_element->second->mu = (exist_element->second->sigma2*new_mu 
                                                  + new_sigma2*exist_element->second->mu)/(exist_element->second->sigma2 + new_sigma2);
                        exist_element->second->sigma2 = exist_element->second->sigma2*new_sigma2/(exist_element->second->sigma2+new_sigma2);
                    }
                    else
                    {
                        Feature *new_ftr = new Feature(frame.get(), px_found, 0);
                        Seed *new_seed = new Seed(new_ftr, 1.0, 1.0/it->second->z_range); //don't care depth mean, it will be overwritten. 
                                                                                            // z_range is the same with ref
                        new_ftr->type = Feature::EDGELET;
                        // compute depth in cur frame
                        new_seed->mu = new_mu;
                        // uncertainty in cur frame
                        new_seed->sigma2 = new_sigma2;
                        new_seed->batch_id = -1;    // -1 means a temp seed(for propagation), will be deleted. TODO: is there a better way?
                        frame->depth_map_.insert(make_pair(px_ipos[0] + px_ipos[1]*frame->cam_->width(), new_seed));
                    }
                }
                ++it;
            }
            active_keyframe_->depth_map_quality_ = good_edge;
        }
#ifdef SVO_TRACE
        permon_.stopTimer("depth_map_update");
        permon_.log("update_n_edge", n_updated);
        permon_.writeToFile();
#endif
        if (options_.verbose)
        SVO_INFO_STREAM("update depth map success");
    }

    void DepthMapManager::clearFrameQueue()
    {
        while (!frame_queue_.empty())
            frame_queue_.pop();
    }

}
