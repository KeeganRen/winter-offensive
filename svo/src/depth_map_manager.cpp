#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_map_manager.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>

namespace svo {
    DepthMapManager::DepthMapManager(feature_detection::DetectorPtr edge_detector, Map& map) :
    edge_detector_(edge_detector),
    thread_(NULL),
    depth_map_updating_halt_(false),
    map_(map),
    new_keyframe_set_(false),
    new_keyframe_mean_depth_(0.0),
    new_keyframe_min_depth_(0.0)
    {}

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
                    frame_queue_.pop();
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

    void DepthMapManager::initializeDepthMap(FramePtr frame)
    {
        Features new_features;
        edge_detector_->detect(frame.get(), frame->img_pyr_,
                0.0, new_features);
        depth_map_updating_halt_ = true;
        {
            lock_t lock(frame->depth_map_mut_);
            frame->clearDepthMap();
            std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
                    frame->depth_map_.insert(make_pair(ftr->px[0]+ftr->px[1]*frame->cam_->width(), new PixelDepthHypothesis(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_)));
                    });
        }
        if (active_keyframe_ != NULL) // do propagation
        {

        }
        active_keyframe_ = frame;
    }

    void DepthMapManager::updateMap(FramePtr frame)
    {
        lock_t lock(frame->depth_map_mut_);
        // access to depth map
        for (int i=100; i>0; i--);
    }

    void DepthMapManager::clearFrameQueue()
    {
        while (!frame_queue_.empty())
            frame_queue_.pop();
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
}
