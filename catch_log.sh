#if the first arg is watch then execute adb logcat -s adsprpc
if [ "$1" = "watch" ]; then
    echo "Watching adsprpc logs (Ctrl-C to stop)..."
    adb logcat -s adsprpc -d
    exit $?
fi
# catch log and store with filename with timestamp.
adb logcat -d | grep "htp_ops_test" > "htp_ops_test_htp_ops_test_$(date +%Y%m%d_%H%M%S).log"
adb logcat -d | grep -E ' [IVDWE] adsprpc : ' > "htp_ops_test_adsprpc_$(date +%Y%m%d_%H%M%S).log"
adb logcat -d > "htp_ops_test_all_$(date +%Y%m%d_%H%M%S).log"