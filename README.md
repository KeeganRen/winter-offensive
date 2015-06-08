SVO
===

This code implements a semi-direct monocular visual odometry pipeline.

Video: http://youtu.be/2YnIMfw6bJY

Paper: http://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf

#### Disclaimer

SVO has been tested under ROS Groovy, Hydro and Indigo with Ubuntu 12.04, 13.04 and 14.04. This is research code, any fitness for a particular purpose is disclaimed.


#### Licence

The source code is released under a GPLv3 licence. A closed-source professional edition is available for commercial purposes. In this case, please contact the authors for further info.


#### Citing

If you use SVO in an academic context, please cite the following publication:

    @inproceedings{Forster2014ICRA,
      author = {Forster, Christian and Pizzoli, Matia and Scaramuzza, Davide},
      title = {{SVO}: Fast Semi-Direct Monocular Visual Odometry},
      booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
      year = {2014}
    }
    
    
#### Documentation

The API is documented here: http://uzh-rpg.github.io/rpg_svo/doc/

#### Instructions

See the Wiki for more instructions. https://github.com/uzh-rpg/rpg_svo/wiki

#### Contributing

You are very welcome to contribute to SVO by opening a pull request via Github.
I try to follow the ROS C++ style guide http://wiki.ros.org/CppStyleGuide

#### Known issue
1. After following the installation directions exactly, the below error occurs when running 'catkin_make' to build the vikit and svo packages. The 'IS_ARM' flag is not set. I'm running Ubuntu 12.04 on a x86_64 machine (i7).

  Any ideas?
```
rpg_vikit/vikit_common/CMakeFiles/vikit_common.dir/src/pinhole_camera.cpp.o
[ 42%] Building CXX object rpg_vikit/vikit_common/CMakeFiles/vikit_common.dir/src/homography.cpp.o
[ 44%] Building CXX object rpg_vikit/vikit_common/CMakeFiles/vikit_common.dir/src/img_align.cpp.o
/tmp/ccXXH67w.s: Assembler messages:
/tmp/ccXXH67w.s:481: Error: no such instruction: vfmadd312ss .LC0(%rip),%xmm0,%xmm0' make[2]: *** [rpg_vikit/vikit_common/CMakeFiles/vikit_common.dir/src/robust_cost.cpp.o] Error 1 make[2]: *** Waiting for unfinished jobs.... /tmp/ccTKsebv.s: Assembler messages: /tmp/ccTKsebv.s:367: Error: no such instruction:vfmadd312sd 8(%rsi),%xmm5,%xmm1'
/tmp/ccTKsebv.s:369: Error: no such instruction: vfmadd312sd 16(%rsi),%xmm5,%xmm2' /tmp/ccTKsebv.s:370: Error: no such instruction:vfmadd312sd (%rsi),%xmm5,%xmm0'
...
/tmp/cc4xDiOp.s:21241: Error: no such instruction: vfmadd312sd (%rax,%r15,8),%xmm1,%xmm7' /tmp/cc4xDiOp.s:21244: Error: no such instruction:vfmadd312sd (%rax,%rsi,8),%xmm1,%xmm8'
make[2]: *** [rpg_vikit/vikit_common/CMakeFiles/vikit_common.dir/src/homography.cpp.o] Error 1
make[1]: *** [rpg_vikit/vikit_common/CMakeFiles/vikit_common.dir/all] Error 2
make: *** [all] Error 2
Invoking "make" failed
```
  
  the reason is most likely that a very recent core i7 CPU and an outdated   c++ compiler (such as g++ version 4.6 that comes with Ubuntu 12.04.) are   used together.
  There are several ways to deal with this problem, but the most simple one   is to replace all "-march=native" flags in all CMakeLists.txt files with   "-march=corei7" since the new CPUs are not correctly identified automatically by the old compiler for the highest level of code optimization that SVO requests.
