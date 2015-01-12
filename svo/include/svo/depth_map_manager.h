#ifndef SVO_DEPTH_MAP_MANAGER_H_
#define SVO_DEPTH_MAP_MANAGER_H_

#include <queue>
#include <boost/thread.hpp>
#include <svo/feature_detection.h>
#include <svo/global.h>
#include <svo/matcher.h>
#include <vikit/performance_monitor.h>

namespace svo {
    class Map;
    class Frame;
    
    class DepthMapManager
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            typedef boost::unique_lock<boost::mutex> lock_t;
            struct Options
            {
                int maintain_edges; // number of edge point to maintian
                double max_depth_var_for_tracking;  //maximal depth variance that can be incoperated in tracking
                bool verbose;
                Options():
                    maintain_edges(800),
                    max_depth_var_for_tracking(150.0),
                    verbose(false)
                {}
            } options_;

            DepthMapManager(
                    feature_detection::DetectorPtr edge_detector,
                    Map& map);
            virtual ~DepthMapManager();

            void startThread();
            void stopThread();
            void addFrame(FramePtr frame);
            void addKeyframe(FramePtr frame, double depth_mean, double depth_min);
            void removeKeyframe(FramePtr frame);
            void reset();
            FramePtr getActiveKeyframe()
            {
                return active_keyframe_;
            }

        protected:
            FramePtr active_keyframe_;  // active keyframe that currently used as reference
            feature_detection::DetectorPtr edge_detector_;  
            boost::thread* thread_;
            bool depth_map_updating_halt_;
            std::queue<FramePtr> frame_queue_;
            boost::mutex frame_queue_mut_;
            boost::condition_variable frame_queue_cond_;
            Matcher matcher_;
            Map& map_;

            FramePtr new_keyframe_;
            bool new_keyframe_set_;
            double new_keyframe_mean_depth_;
            double new_keyframe_min_depth_;

            vk::PerformanceMonitor permon_;

            // initialize the depth map of the input frame
            // if has active keyframe, propagate the depth map
            // else perform random init
            void initializeDepthMap(FramePtr frame);

            // refine current depth map by stereo matching between the active keyframe and the new one
            void updateMap(FramePtr frame);

            void clearFrameQueue();

            void updateDMapLoop();
    };
} // namespace svo

#endif // SVO_DEPTH_MAP_MANAGER_H_

