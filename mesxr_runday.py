#!/usr/bin/env python
"""
"""
import os
import numpy as np
import mesxr.operation.camera as cam

base_dir = '/home/det/meSXR/configurations'
test = False

# Print introductory message
print('ME-SXR runday startup script.')
print('Version 1.0')
print('Code by Patrick VanMeter, UW-Madison')

# Determine the desired energy range
calib_options = [f for f in os.listdir(base_dir) if not f.startswith('.')]
calib_options.sort()
calib_descriptions = [None for x in range(len(calib_options))]
for index, opt in enumerate(calib_options):
	try:
		with open(os.path.join(base_dir, opt, '.description.txt'), 'r') as f:
			calib_descriptions[index] = f.read().rstrip()
	except:
		calib_descriptions[index] = 'Description not available.'

print('\nAvailable energy ranges')

for index in range(len(calib_options)):
	print('{0:d}) {1:}: {2:}'.format(index, calib_options[index], calib_descriptions[index]))

user_input = raw_input('Select an energy range: ')
cal_index = int(user_input)
calib_choice = calib_options[cal_index]
calib_dir = os.path.join(base_dir, calib_choice)

# Determine the specific configuration
config_options = [f for f in os.listdir(calib_dir) if not f.startswith('.')]
config_options.sort()
config_descriptions = [None for x in range(len(config_options))]
for index, opt in enumerate(config_options):
	try:
		with open(os.path.join(calib_dir, opt, '.description.txt'), 'r') as f:
			config_descriptions[index] = f.read().rstrip()
	except:
		config_descriptions[index] = 'Description not available.'

print('\nAvailable configurations:')

for index in range(len(config_options)):
	print('{0:d}) {1:}: {2:}'.format(index, config_options[index], config_descriptions[index]))

user_input = raw_input('Select a configuration: ')
config_index = int(user_input)
config_choice = config_options[config_index]

# Determine the camera settings
exp_time = 0.001
exp_period = 0.002
delay = 0
n_frames = 30

print('\nDefault camera settings:')
print('exp_time={0:} sec., exp_period={1:} sec., delay={2:} sec., n_frames={3:}'.format(exp_time, exp_period, delay, n_frames))
user_input = raw_input('Are these settings OK? (y/n): ') or 'y'

if user_input[0].lower() == 'y':
	print('Using default values.')

elif user_input[0].lower() == 'n':
	while user_input[0].lower() != 'y':
		print('Please input new settings. Leave empty to keep default value.')
		exp_time = float(raw_input('exp_time (seconds): ') or exp_time)
		exp_period = float(raw_input('exp_period (seconds): ') or exp_period)
		delay = float(raw_input('delay (seconds): ') or delay)
		n_frames = int(raw_input('n_frames: ') or n_frames)

		print('New camera settings:')
		print('exp_time={0:} sec., exp_period={1:} sec., delay={2:} sec., n_frames={3:}'.format(exp_time, exp_period, delay, n_frames))
		user_input = raw_input('Are these settings OK? (y/n): ') or 'y'

else:
	print('Input not recognized. The default settings will be used.')

# Arm the detector
print('\nArming the detector.')
if test:
	print("cam.arm_detector(0, mode='cycle', write_mds=True, calibration_name={0:}, config_name={1:}, n_frames={2:}, exp_period={3:}, exp_time={4:}, delay={5:})".format(calib_choice, config_choice, n_frames, exp_period, exp_time, delay))
else:
	cam.arm_detector(mode='cycle', write_mds=True, rate_corr=False, calibration_name=calib_choice, config_name=config_choice, n_frames=n_frames, exp_period=exp_period, exp_time=exp_time, delay=delay)

