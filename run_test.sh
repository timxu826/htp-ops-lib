persona=xujia
# Base directory on device
basedir=/data/local/tmp/$persona/llama.cpp
htp_ops_lib_dir=/data/local/tmp/$persona/htp-ops-lib
hexagon_tool_dir=$htp_ops_lib_dir/hexagon_ReleaseG_toolv19_v75
android_release_dir=$htp_ops_lib_dir/android_ReleaseG_aarch64

adb shell "touch $hexagon_tool_dir/htp_ops_test.farf"
adb shell "logcat -c"
adb shell "cd $htp_ops_lib_dir && FASTRPC_PERF_ADSP=1 FASTRPC_PERF_KERNEL=1 FASTRPC_PERF_FREQ=1 LD_LIBRARY_PATH=$android_release_dir ADSP_LIBRARY_PATH=$hexagon_tool_dir DSP_LIBRARY_PATH=$hexagon_tool_dir $android_release_dir/htp_ops_test"
# adb shell "FASTRPC_PERF_ADSP=1 FASTRPC_PERF_KERNEL=1 FASTRPC_PERF_FREQ=1 LD_LIBRARY_PATH=$basedir/lib:$android_release_dir ADSP_LIBRARY_PATH=$basedir/lib:$hexagon_tool_dir $android_release_dir/htp_ops_test"