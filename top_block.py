#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: MSK144_tests
# Author: jeff millar
# Copyright: GPL V3 2026
# Description: testing wideband decoder for msk144
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.filter import pfb
import sip



class top_block(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "MSK144_tests", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("MSK144_tests")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "top_block")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 48000
        self.taps = taps = firdes.low_pass(1.0, samp_rate, 3000, 1500, 1, 6.76)

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            2048, #size
            window.WIN_HAMMING, #wintype
            0, #fc
            48000, #bw
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(True)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-80, -40)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_win)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_freq_sink_x_1 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_1.set_update_time(0.10)
        self.qtgui_freq_sink_x_1.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_1.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_1.enable_autoscale(False)
        self.qtgui_freq_sink_x_1.enable_grid(False)
        self.qtgui_freq_sink_x_1.set_fft_average(1.0)
        self.qtgui_freq_sink_x_1.enable_axis_labels(True)
        self.qtgui_freq_sink_x_1.enable_control_panel(False)
        self.qtgui_freq_sink_x_1.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_1_win = sip.wrapinstance(self.qtgui_freq_sink_x_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_1_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            12000, #bw
            'ch15', #name
            4,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(4):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.pfb_channelizer_ccf_0 = pfb.channelizer_ccf(
            48,
            taps,
            (48/4),
            100)
        self.pfb_channelizer_ccf_0.set_channel_map([])
        self.pfb_channelizer_ccf_0.declare_sample_delay(0)
        self.blocks_wavfile_source_0 = blocks.wavfile_source('/home/jeff/ham/mapmsk144/MSK144/simulations/msk144_20260322_084647_10sig.wav', False)
        self.blocks_null_sink_0_5 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_3_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_3_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_3 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_0_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_3 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_1_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_1_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_0_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_3 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_1_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_1_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_0_1_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_0_1_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_0_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_float_to_complex_0, 0), (self.pfb_channelizer_ccf_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.qtgui_freq_sink_x_1, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.blocks_wavfile_source_0, 1), (self.blocks_float_to_complex_0, 1))
        self.connect((self.blocks_wavfile_source_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_wavfile_source_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_wavfile_source_0, 1), (self.qtgui_time_sink_x_0, 1))
        self.connect((self.pfb_channelizer_ccf_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 1), (self.blocks_null_sink_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 4), (self.blocks_null_sink_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 10), (self.blocks_null_sink_0_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 13), (self.blocks_null_sink_0_0_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 29), (self.blocks_null_sink_0_0_0_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 44), (self.blocks_null_sink_0_0_0_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 46), (self.blocks_null_sink_0_0_0_0_0_1_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 47), (self.blocks_null_sink_0_0_0_0_0_1_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 26), (self.blocks_null_sink_0_0_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 41), (self.blocks_null_sink_0_0_0_0_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 20), (self.blocks_null_sink_0_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 35), (self.blocks_null_sink_0_0_0_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 7), (self.blocks_null_sink_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 23), (self.blocks_null_sink_0_0_1_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 38), (self.blocks_null_sink_0_0_1_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 32), (self.blocks_null_sink_0_0_3, 0))
        self.connect((self.pfb_channelizer_ccf_0, 2), (self.blocks_null_sink_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 5), (self.blocks_null_sink_0_1_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 11), (self.blocks_null_sink_0_1_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 14), (self.blocks_null_sink_0_1_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 30), (self.blocks_null_sink_0_1_0_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 45), (self.blocks_null_sink_0_1_0_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 27), (self.blocks_null_sink_0_1_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 42), (self.blocks_null_sink_0_1_0_0_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 21), (self.blocks_null_sink_0_1_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 36), (self.blocks_null_sink_0_1_0_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 8), (self.blocks_null_sink_0_1_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 24), (self.blocks_null_sink_0_1_1_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 39), (self.blocks_null_sink_0_1_1_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 33), (self.blocks_null_sink_0_1_3, 0))
        self.connect((self.pfb_channelizer_ccf_0, 3), (self.blocks_null_sink_0_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 9), (self.blocks_null_sink_0_2_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 12), (self.blocks_null_sink_0_2_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 28), (self.blocks_null_sink_0_2_0_0_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 43), (self.blocks_null_sink_0_2_0_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 25), (self.blocks_null_sink_0_2_0_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 40), (self.blocks_null_sink_0_2_0_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 19), (self.blocks_null_sink_0_2_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 34), (self.blocks_null_sink_0_2_2, 0))
        self.connect((self.pfb_channelizer_ccf_0, 6), (self.blocks_null_sink_0_3, 0))
        self.connect((self.pfb_channelizer_ccf_0, 22), (self.blocks_null_sink_0_3_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 37), (self.blocks_null_sink_0_3_1, 0))
        self.connect((self.pfb_channelizer_ccf_0, 31), (self.blocks_null_sink_0_5, 0))
        self.connect((self.pfb_channelizer_ccf_0, 16), (self.qtgui_freq_sink_x_0, 1))
        self.connect((self.pfb_channelizer_ccf_0, 18), (self.qtgui_freq_sink_x_0, 3))
        self.connect((self.pfb_channelizer_ccf_0, 15), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.pfb_channelizer_ccf_0, 17), (self.qtgui_freq_sink_x_0, 2))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "top_block")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_taps(firdes.low_pass(1.0, self.samp_rate, 3000, 1500, 1, 6.76))
        self.qtgui_freq_sink_x_1.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps
        self.pfb_channelizer_ccf_0.set_taps(self.taps)




def main(top_block_cls=top_block, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
