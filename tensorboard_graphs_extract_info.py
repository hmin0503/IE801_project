import tensorflow as tf
import os

log_dir = './a2c_atari_tensorboard/A2C_1'  # Path to your TensorBoard logs

# List the event files in the log directory
event_files = [os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.startswith('events.out.tfevents')]

# Create a summary iterator to read the event files
for event_file in event_files:
    for summary in tf.compat.v1.train.summary_iterator(event_file):
        for v in summary.summary.value:
            # Adjust tags as needed
            if v.tag == 'rollout/ep_rew_mean':  # For mean reward
                print(f"Step: {summary.step}, Mean Reward: {v.simple_value}")
            elif v.tag == 'rollout/ep_len_mean':  # For mean episode length
                print(f"Step: {summary.step}, Mean Episode Length: {v.simple_value}")
