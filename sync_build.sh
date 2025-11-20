# LD_LIBRARY_PATH=/data/local/tmp/xujia/htp-ops-lib/android_ReleaseG_aarch64 ADSP_LIBRARY_PATH=/data/local/tmp/xujia/htp-ops-lib/hexagon_ReleaseG_toolv19_v75 ./htp_ops_test
# remove android_ReleaseG_aarch64 and hexagon_ReleaseG_toolv19_v75 on device first
LIB_PATH=/data/local/tmp/xujia/htp-ops-lib
ANDROID_PATH=$LIB_PATH/android_ReleaseG_aarch64
HEXAGON_PATH=$LIB_PATH/hexagon_ReleaseG_toolv19_v75

adb shell rm -rf $ANDROID_PATH
adb shell rm -rf $HEXAGON_PATH
# push new build files
adb push ./android_ReleaseG_aarch64 $ANDROID_PATH
adb push ./hexagon_ReleaseG_toolv19_v75 $HEXAGON_PATH
# adb push ./htp_ops_test $LIB_PATH/htp_ops_test
adb shell chmod +x $ANDROID_PATH/htp_ops_test

# debug files 
adb push htp_ops_test.debugconfig $ANDROID_PATH/
adb push htp_ops_test.debugconfig $HEXAGON_PATH/

adb shell LD_LIBRARY_PATH=$ANDROID_PATH ADSP_LIBRARY_PATH=$HEXAGON_PATH $ANDROID_PATH/htp_ops_test
