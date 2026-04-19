from map144_app.capture import load_capture
from map144_app.analysis_engine import AnalysisEngine

iq, meta = load_capture('MSK144/captures/20260402_165412Z_capture_radio_50260kHz.wav')
eng = AnalysisEngine(iq, meta)
results = eng.run_replay(nb_factor=5.7, nb_enabled=True)

print("spectrogram shape:", results['spectrogram_data'].shape)
print("ch_snr shape:     ", results['ch_snr_history'].shape)
print("markers:          ", len(results['jt9_markers']))
print("blanked blocks:   ", len(results['blanked_times_s']))
print("duration:         ", results['duration_secs'], "s")
