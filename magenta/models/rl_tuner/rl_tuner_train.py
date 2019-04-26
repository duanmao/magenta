# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Code to train a MelodyQ model.

To run this code on your local machine:
python magenta/models/rl_tuner/rl_tuner_train.py \
--note_rnn_checkpoint_dir 'path' --midi_primer 'primer.mid' \
--training_data_path 'path.tfrecord'
"""
import os
import numpy as np

from magenta.models.rl_tuner import rl_tuner
from magenta.models.rl_tuner import rl_tuner_ops
import matplotlib
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Need to use 'Agg' option for plotting and saving files from command line.
# Can't use 'Agg' in RL Tuner because it breaks plotting in notebooks.
# pylint: disable=g-import-not-at-top,wrong-import-position
matplotlib.use('Agg')

# pylint: enable=g-import-not-at-top,wrong-import-position


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '',
                           'Directory where the model will save its'
                           'compositions and checkpoints (midi files)')
tf.app.flags.DEFINE_string('note_rnn_checkpoint_dir', '',
                           'Path to directory holding checkpoints for note rnn'
                           'melody prediction models. These will be loaded into'
                           'the NoteRNNLoader class object. The directory '
                           'should contain a train subdirectory')
tf.app.flags.DEFINE_string('note_rnn_checkpoint_name', 'note_rnn.ckpt',
                           'Filename of a checkpoint within the '
                           'note_rnn_checkpoint_dir directory.')
tf.app.flags.DEFINE_string('note_rnn_type', 'default',
                           'If `default`, will use the basic LSTM described in '
                           'the research paper. If `basic_rnn`, will assume '
                           'the checkpoint is from a Magenta basic_rnn model.')
tf.app.flags.DEFINE_string('midi_primer', './testdata/primer.mid',
                           'A midi file that can be used to prime the model')
tf.app.flags.DEFINE_integer('training_steps', 1000000,
                            'The number of steps used to train the model')
tf.app.flags.DEFINE_integer('exploration_steps', 500000,
                            'The number of steps over which the models'
                            'probability of taking a random action (exploring)'
                            'will be annealed from 1.0 to its normal'
                            'exploration probability. Typically about half the'
                            'training_steps')
tf.app.flags.DEFINE_string('exploration_mode', 'egreedy',
                           'Can be either egreedy for epsilon-greedy or '
                           'boltzmann, which will sample from the models'
                           'output distribution to select the next action')
tf.app.flags.DEFINE_integer('output_every_nth', 50000,
                            'The number of steps before the model will evaluate'
                            'itself and store a checkpoint')
tf.app.flags.DEFINE_integer('num_notes_in_melody', 32,
                            'The number of notes in each composition')
tf.app.flags.DEFINE_float('reward_scaler', 0.5,
                          'The weight placed on music theory rewards')
tf.app.flags.DEFINE_string('training_data_path', '',
                           'Directory where the model will get melody training'
                           'examples')
tf.app.flags.DEFINE_string('algorithm', 'q',
                           'The name of the algorithm to use for training the'
                           'model. Can be q, psi, or g')
tf.app.flags.DEFINE_string('running_mode', 'default',
                           'the purpose of the current run')

def run_algorithm(alg, dqn_hparams, hparams, num_compositions):
  output_dir = os.path.join(FLAGS.output_dir, alg)
  output_ckpt = alg + '.ckpt'
  backup_checkpoint_file = os.path.join(FLAGS.note_rnn_checkpoint_dir,
                                        FLAGS.note_rnn_checkpoint_name)
  rlt = rl_tuner.RLTuner(output_dir,
                         midi_primer=FLAGS.midi_primer,
                         dqn_hparams=dqn_hparams,
                         reward_scaler=FLAGS.reward_scaler,
                         save_name=output_ckpt,
                         output_every_nth=FLAGS.output_every_nth,
                         note_rnn_checkpoint_dir=FLAGS.note_rnn_checkpoint_dir,
                         note_rnn_checkpoint_file=backup_checkpoint_file,
                         note_rnn_type=FLAGS.note_rnn_type,
                         note_rnn_hparams=hparams,
                         num_notes_in_melody=FLAGS.num_notes_in_melody,
                         exploration_mode=FLAGS.exploration_mode,
                         algorithm=alg)

  tf.logging.info('Saving images and melodies to: %s', rlt.output_dir)

  if (alg != "none_rl"):
    tf.logging.info('Training...')
    rlt.train(num_steps=FLAGS.training_steps,
              exploration_period=FLAGS.exploration_steps)

  tf.logging.info('Finished training. Saving output figures and composition.')
  rlt.plot_rewards(image_name='Rewards-' + alg + '.eps')

  rlt.generate_music_sequence(visualize_probs=True, title=alg,
                              prob_image_name=alg + '.png')

  rlt.save_model_and_figs(alg)

  tf.logging.info('Calculating music theory metric stats for %d '
                  'compositions.', num_compositions)
  rlt.evaluate_music_theory_metrics(num_compositions=num_compositions)
  return rlt

def plot_comparison_figure(x, y1, y2, start_index, output_dir, name, alg1="Q",                             alg2="RL only"):
  """Plots the specific type of rewards received as the models were evaluated during training.

  Args:
    y1: rewards that the first model received
    y2: rewards that the second rl_tuner model received
    start_at_epoch: Training epoch where the plot should begin.
    name: name of this figure
    alg1: algorithm of the first model
    alg2: algorithm of the second model
  """

  plt.figure()
  plt.plot(x[start_index:], y1[start_index:])
  plt.plot(x[start_index:], y2[start_index:])
  plt.xlabel('Training epoch')
  plt.ylabel('Averaged reward over 100 compositions')
  plt.legend([alg1, alg2], loc='best')
  plt.savefig(output_dir + '/' + 'EvaluationRewards-' + alg1 + "-vs-" + alg2 +              '-' + name + '.eps',)

def plot_comparison(rlt1, rlt2, start_at_epoch=0):
  """Plots the rewards received as the models were evaluated during training.

  Args:
    rlt1: the first rl_tuner model to compare
    rlt2: the second rl_tuner model to compare
    start_at_epoch: Training epoch where the plot should begin.
  """

  reward_batch = rlt1.output_every_nth
  x = [reward_batch * i for i in np.arange(len(rlt1.eval_avg_reward))]
  start_index = int(start_at_epoch / rlt1.output_every_nth)
  plot_comparison_figure(x, rlt1.eval_avg_reward, rlt2.eval_avg_reward,                                start_index, rlt1.output_dir, 'Rewards')
  plot_comparison_figure(x, rlt1.eval_avg_music_theory_reward,                                         rlt2.eval_avg_music_theory_reward, start_index,                               rlt1.output_dir, 'MusicTheoryRewards')
  plot_comparison_figure(x, rlt1.eval_avg_note_rnn_reward,                                             rlt2.eval_avg_note_rnn_reward, start_index,                                   rlt1.output_dir, 'NoteRNNRewards')

def main(_):
  if FLAGS.note_rnn_type == 'basic_rnn':
    hparams = rl_tuner_ops.basic_rnn_hparams()
  else:
    hparams = rl_tuner_ops.default_hparams()

  dqn_hparams = tf.contrib.training.HParams(random_action_probability=0.1,
                                            store_every_nth=1,
                                            train_every_nth=5,
                                            minibatch_size=32,
                                            discount_rate=0.5,
                                            max_experience=100000,
                                            target_network_update_rate=0.01)

  num_compositions = 100000
  if (FLAGS.running_mode == 'comparison'):
    tf.logging.info("Running for comparison")
    # num_compositions = 1000

  defaultRlt = run_algorithm(FLAGS.algorithm, dqn_hparams, hparams,                                        num_compositions)
  if (FLAGS.running_mode == 'comparison'):
    pureRlt = run_algorithm('pure_rl', dqn_hparams, hparams, num_compositions)
    plot_comparison(defaultRlt, pureRlt)

def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
