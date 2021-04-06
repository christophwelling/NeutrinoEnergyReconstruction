import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.cm
from NuRadioReco.utilities import units, fft, bandpass_filter
import NuRadioMC.SignalGen.askaryan
import NuRadioReco.detector.antennapattern
import NuRadioReco.detector.RNO_G.analog_components
import NuRadioMC.utilities.medium
import numpy as np

energy = 1.e17 * units.eV
samples = 512
sampling_rate = 2. * units.GHz
max_viewing_angle = 5
viewing_angles = np.arange(0, max_viewing_angle + 1.e-3, 1.) * units.deg
max_receiving_angle = 160
receiving_angles = np.arange(90., max_receiving_angle + 1.e-3, 10.) * units.deg
ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
ior = ice.get_index_of_refraction([0, 0, -1000])
cherenkov_angle = np.arccos(1. / ior)
freqs = np.fft.rfftfreq(samples, 1. / sampling_rate)
times = np.arange(samples) / sampling_rate

cmap = plt.get_cmap('Blues')

antenna_provier = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
antenna_pattern = antenna_provier.load_antenna_pattern('vpol_4inch_center')
antenna_response = antenna_pattern.get_antenna_response_vectorized(
    freqs,
    90. * units.deg,
    0.,
    0.,
    0.,
    90. * units.deg,
    90. * units.deg
)
amp_response = NuRadioReco.detector.RNO_G.analog_components.load_amp_response('iglu')

fig1 = plt.figure(figsize=(6, 8))
ax1_1 = fig1.add_subplot(211)
ax1_1.set_facecolor('silver')
ax1_2 = fig1.add_subplot(212)
ax1_2.set_facecolor('silver')
for i_angle, viewing_angle in enumerate(viewing_angles):
    efield_freq = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
        energy=energy,
        theta=cherenkov_angle + viewing_angle,
        dt=1. / sampling_rate,
        shower_type='HAD',
        n_index=ior,
        N=samples,
        R=2. * units.km,
        model='Alvarez2009'
    )
    line_color = cmap(viewing_angle/ (max_viewing_angle * units.deg))
    voltage_spectrum = efield_freq * antenna_response['theta'] * amp_response['gain'](freqs) * amp_response['phase'](freqs) * bandpass_filter.get_filter_response(
        freqs,
        [.1, .3],
        'butter',
        10
    )
    voltage_trace = fft.freq2time(voltage_spectrum, sampling_rate)
    ax1_1.plot(
        times,
        voltage_trace / units.mV,
        color=line_color,
        alpha=1.
    )

for i_angle, receiving_angle in enumerate(receiving_angles):
    efield_freq = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
        energy=energy,
        theta=cherenkov_angle + 1. * units.deg,
        dt=1. / sampling_rate,
        shower_type='HAD',
        n_index=ior,
        N=samples,
        R=1. * units.km,
        model='Alvarez2009'
    )
    line_color = cmap((receiving_angle - 90. * units.deg) / ((max_receiving_angle - 90) * units.deg))
    antenna_response = antenna_pattern.get_antenna_response_vectorized(
        freqs,
        receiving_angle,
        0.,
        0.,
        0.,
        90. * units.deg,
        90. * units.deg
    )
    voltage_spectrum = efield_freq * antenna_response['theta'] * amp_response['gain'](freqs) * amp_response['phase'](freqs) * bandpass_filter.get_filter_response(
        freqs,
        [.1, .3],
        'butter',
        10
    )
    voltage_trace = fft.freq2time(voltage_spectrum, sampling_rate)
    ax1_2.plot(
        times,
        voltage_trace / units.mV,
        color=line_color,
        alpha=1.
    )
viewing_angle_norm = matplotlib.colors.Normalize(vmin=0, vmax=max_viewing_angle)
viewing_angle_cmap = matplotlib.cm.ScalarMappable(norm=viewing_angle_norm, cmap=plt.get_cmap('Blues'))
plt.colorbar(viewing_angle_cmap, ax=ax1_1).set_label(r'$\phi - \phi_{Cherenkov} [^\circ]$')
receiving_angle_norm = matplotlib.colors.Normalize(vmin=90, vmax=max_receiving_angle)
receiving_angle_cmap = matplotlib.cm.ScalarMappable(norm=receiving_angle_norm, cmap=plt.get_cmap('Blues'))
plt.colorbar(receiving_angle_cmap, ax=ax1_2).set_label(r'$\theta_r [^\circ]$')
ax1_1.grid()
ax1_1.set_xlim([130, 200])
ax1_1.set_xlabel('t [ns]')
ax1_1.set_ylabel('U [mV]')
ax1_2.grid()
ax1_2.set_xlim([130, 200])
ax1_2.set_xlabel('t [ns]')
ax1_2.set_ylabel('U [mV]')
fig1.tight_layout()
fig1.savefig('plots/pulse_shape.png')
