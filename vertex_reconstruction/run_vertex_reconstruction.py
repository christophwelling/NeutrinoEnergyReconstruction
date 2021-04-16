import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelPulseFinderSimulator
import NuRadioReco.detector.generic_detector
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelTimeOffsetCalculator
import NuRadioReco.modules.neutrinoVertexReconstructor.neutrino3DVertexReconstructor
import NuRadioReco.modules.channelSignalPropertiesFromNeighbors
import NuRadioReco.modules.iftElectricFieldReconstructor.iftElectricFieldReconstructor
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
from NuRadioReco.utilities import units, bandpass_filter
import NuRadioMC.utilities.medium
import NuRadioReco.framework.base_trace
from NuRadioReco.framework.parameters import channelParameters as chp
import glob
import argparse

noise_level = 10. * units.mV
sampling_rate = 5. * units.GHz
# plot_channel_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
vertex_channel_ids = [0, 3, 4, 5, 6, 9, 10, 12, 13]
# vertex_channel_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13]
vertex_reco_passband = [.13, .3]
efield_reco_passband = [.1, .5]
ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
signal_region_colors = ['red', 'green', 'blue']

event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin(glob.glob('../event_sim/data/*.nur'))

pulse_finder = NuRadioReco.modules.channelPulseFinderSimulator.channelPulseFinderSimulator()
pulse_finder.begin(
    noise_level=noise_level,
    min_snr=2.5,
    signal_window_limits=(-10, 40)
)


noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channel_resampler = NuRadioReco.modules.channelResampler.channelResampler()

det = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename='../energy_reconstruction/RNO_station.json',
    default_station=101,
    default_channel=0,
    antenna_by_depth=False
)

spec = np.ones(int(128 * sampling_rate + 1)) * bandpass_filter.get_filter_response(np.fft.rfftfreq(int(256 * sampling_rate), 1. / sampling_rate), vertex_reco_passband, 'butter', 10)
efield_template = NuRadioReco.framework.base_trace.BaseTrace()
efield_template.set_frequency_spectrum(spec, sampling_rate)
efield_template.apply_time_shift(20. * units.ns, True)

vertex_reconstructor = NuRadioReco.modules.neutrinoVertexReconstructor.neutrino3DVertexReconstructor.neutrino3DVertexReconstructor(
    'lookup_tables'
)
vertex_reconstructor.begin(
    station_id=101,
    channel_ids=vertex_channel_ids,
    detector=det,
    template=efield_template,
    distances_2d=np.arange(100, 3900, 50),
    z_coordinates_2d=np.arange(-2700, -100, 50),
    azimuths_2d=np.arange(0, 360.1, 5.) * units.deg,
    distance_step_3d=5,
    z_step_3d=5,
    widths_3d=np.arange(-100, 100, 5),
    passband=vertex_reco_passband,
    debug_folder='plots/vertex_reco'
)

for i_event, event in enumerate(event_reader.run()):
    if i_event >= 100000:
        break
    if i_event <= 1:
        continue
    print('Event {}, {}, {}'.format(i_event, event.get_run_number(), event.get_id()))
    station = event.get_station(101)
    station.set_is_neutrino()
    channel_resampler.run(event, station, det, 2.)
    sim_station = station.get_sim_station()
    noise_adder.run(
        event,
        station,
        det,
        amplitude=noise_level,
        type='rayleigh'
    )
    pulse_finder.run(
        event,
        station,
        det
    )
    plt.close('all')
    fig1 = plt.figure(figsize=(12, 2 * len(vertex_channel_ids)))
    channel_resampler.run(event, station, det, sampling_rate)
    for i_channel, channel_id in enumerate(vertex_channel_ids):
        channel = station.get_channel(channel_id)
        ax1_1 = fig1.add_subplot(len(vertex_channel_ids) // 2 + len(vertex_channel_ids) % 2, 2, i_channel + 1)
        ax1_1.grid()
        ax1_1.plot(channel.get_times(), channel.get_trace() / units.mV)
        sim_channel_sum = None
        for sim_channel in sim_station.get_channels_by_channel_id(channel_id):
            if sim_channel_sum is None:
                sim_channel_sum = sim_channel
            else:
                sim_channel_sum += sim_channel
        if channel.has_parameter(chp.signal_regions):
            for i_region, signal_region in enumerate(channel.get_parameter(chp.signal_regions)):
                ax1_1.axvspan(signal_region[0], signal_region[1], color='r', alpha=.2)
                ax1_1.text(
                    signal_region[0] + 100,
                    -10,
                    'SNR={:.1f}'.format(channel.get_parameter(chp.signal_region_snrs)[i_region]),
                    color='r'
                )
        if sim_channel_sum is not None:
            ax1_1.plot(sim_channel_sum.get_times(), sim_channel_sum.get_trace() / units.mV)
        ax1_1.set_title('Ch. {}'.format(channel_id))
    fig1.tight_layout()
    fig1.savefig('plots/traces/traces_{}_{}.png'.format(event.get_id(), event.get_run_number()))
    vertex_reconstructor.run(event, station, det , True)
