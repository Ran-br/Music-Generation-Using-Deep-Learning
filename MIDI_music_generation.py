import pickle
import pretty_midi as pm
import numpy as np
import os
import random
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Activation, ReLU, TimeDistributed, Bidirectional, Conv1D, MaxPooling1D, Embedding, Flatten
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# Uncomment if you want to make sure you have GPU in your devices
# from tensorflow_core.python.client import device_lib
# print(device_lib.list_local_devices())

# Sampling freq of the columns for piano roll. The higher, the more "timeline" columns we have.
SAMPLING_FREQ = 20
WINDOW_SIZE = 200
VELOCITY_CONST = 64
# The duration of the song we want to be generated (in seconds)
GENERATED_SONG_DURATION = 30
NUM_OF_EPOCHS = 100


def piano_roll_to_pretty_midi(piano_roll, fs=SAMPLING_FREQ, program=0):
    notes, frames = piano_roll.shape
    PM = pm.PrettyMIDI()
    instrument = pm.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pm.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    PM.instruments.append(instrument)
    return PM


def get_piano_roll(instrument, end_time):
    # All instruments get the same "times" parameter in order to make the piano roll timeline the same.
    piano_roll = instrument.get_piano_roll(fs=SAMPLING_FREQ, times=np.arange(0, end_time, 1. / SAMPLING_FREQ))
    return piano_roll


def print_instrument_info(i, instrument):
    instrument_notes_len = len(instrument.notes)
    end_time = instrument.get_end_time()
    print(f"Instrument #{i} - {instrument.name} ({instrument.program}) | {instrument_notes_len} notes | {end_time:.2f} seconds")


def get_time_note_dict(piano_roll):
    times = np.unique(np.where(piano_roll > 0)[1])
    index = np.where(piano_roll > 0)
    dict_keys_time = {}

    for time in times:
        index_where = np.where(index[1] == time)
        notes = index[0][index_where]
        dict_keys_time[time] = notes

    return dict_keys_time
    # return list_of_dict_keys_time


def get_RNN_input_target(notes_list):
    # Creates input, target np arrays in the requested window size
    input_windows = rolling_window([0] * WINDOW_SIZE + notes_list, WINDOW_SIZE)
    target_windows = rolling_window(notes_list, 1)
    input_windows = np.reshape(input_windows, (input_windows.shape[0], input_windows.shape[1], 1))
    return input_windows, target_windows


def rolling_window(a, window):
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def midi_preprocess(path, notes_hash, print_info=False, separate_midi_file=False):
    instruments_piano_roll = {}
    midi_name = path.split('.', 1)[0].split("/")[-1]
    # Load MIDI file into PrettyMIDI object
    midi_data = pm.PrettyMIDI(path)
    end_time = midi_data.get_end_time()
    if print_info:
        print(f"\n---------- {midi_name} | {end_time:.2f} seconds -----------")

    # Separate tracks and print info
    for i, instrument in enumerate(midi_data.instruments):
        # Fix instruments names
        instrument.name = "Drums" if instrument.is_drum else pm.program_to_instrument_name(instrument.program)

        instruments_piano_roll[i] = get_piano_roll(instrument, end_time)

        if print_info:
            print_instrument_info(i, instrument)
        if separate_midi_file:
            # Write MIDI files for each instrument for debugging
            instrument_midi = pm.PrettyMIDI()
            instrument_midi.instruments.append(instrument)
            instrument_midi.write(f'{midi_name}_{i}_instrument.mid')

    all_instruments_piano_roll = get_piano_roll(midi_data, end_time)
    dict_keys_time = get_time_note_dict(all_instruments_piano_roll)

    for key in dict_keys_time.keys():
        dict_keys_time[key] = str(dict_keys_time[key])
        notes_hash.add_new_note(dict_keys_time[key])

    # total time of piano roll, not of the midi file in seconds
    total_time = instruments_piano_roll[0].shape[1]
    notes_list = []
    for time in range(0, total_time, 1):
        if time not in dict_keys_time:
            notes_list += [notes_hash.notes_dict['e']]
        else:
            current_note = dict_keys_time[time]
            notes_list += [notes_hash.notes_dict[current_note]]

    input_windows, target_windows = get_RNN_input_target(notes_list)
    return notes_list, input_windows, target_windows


def compare_real_pred_notes(real_notes_list, pred_notes_list):
    matches_count = 0
    for real, pred in zip(real_notes_list, pred_notes_list):
        if real == pred:
            matches_count += 1
    print("Real song:                   ", real_notes_list)
    print("Pred song (Trained model):   ", pred_notes_list)
    print("Window:                      ", real_notes_list[:WINDOW_SIZE])

    print(f"Length of song's notes is {len(real_notes_list)}")
    print(f"There is {matches_count / len(pred_notes_list) * 100:.2f}% match between Real song and Pred of window size")


def draw_compare_graph(real_input, predicted_input, time):
    plt.scatter(range(time), real_input, c='blue', alpha=0.25)
    plt.scatter(range(time), predicted_input, c='red', alpha=0.5)
    plt.show()


class NotesHash:
    def __init__(self):
        self.notes_dict = {'e': 0}
        self.reversed_notes_dict = {0: 'e'}
        self.token_counter = 1

    def add_new_note(self, new_note):
        if new_note not in self.notes_dict.keys():
            self.notes_dict[new_note] = self.token_counter
            self.reversed_notes_dict[self.token_counter] = new_note
            self.token_counter += 1

    def get_size(self):
        size = len(self.notes_dict)
        return size


class ModelTrainer:
    def __init__(self, files, path, model_arch='lstm', song_epochs=1, epochs=1, batches=128, save_weights=True, save_model=True, save_hash=True, one_hot_input=False):
        self.notes_hash = NotesHash()
        self.songs_epochs = song_epochs
        self.epochs = epochs
        self.batches = batches
        self.files = files
        self.total_songs_num = len(files)
        self.path = path
        self.save_hash = save_hash
        self.save_weights = save_weights
        self.save_model = save_model
        self.all_songs_input_windows = []
        self.all_songs_target_windows = []
        self.one_hot_input = one_hot_input
        self.model = None
        self.model_arch = model_arch

    def preprocess_files(self):
        all_songs_real_notes = []
        temp_all_songs_target_windows = []

        for file in self.files:
            real_notes_list, input_windows, target_windows = midi_preprocess(path=self.path + file, notes_hash=self.notes_hash,
                                                                             print_info=True, separate_midi_file=False)
            self.all_songs_input_windows += [input_windows]
            temp_all_songs_target_windows += [target_windows]
            all_songs_real_notes += [real_notes_list]

        for target in temp_all_songs_target_windows:
            target = to_categorical(target, num_classes=self.notes_hash.get_size())
            self.all_songs_target_windows += [target]

        if self.save_hash:
            self.save_notes_hash()

    def create_model(self):
        # create sequential network, because we are passing activations
        # down the network
        output_layer_size = self.notes_hash.get_size()
        print("output", output_layer_size)

        if self.model_arch == 'lstm':
            # Model 1: Regular LSTM with dense layer on last timeline (not sounds good)
            model = Sequential()
            model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1)))
            model.add(Dense(256))
            model.add(ReLU())
            model.add(Dense(256))
            model.add(ReLU())
            model.add(Dense(output_layer_size, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

        if self.model_arch == 'stacked-lstm':
            # Model 2: Stacked LSTM
            model = Sequential()
            model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1), return_sequences=True))
            model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1)))
            model.add(Dense(512))
            model.add(ReLU())
            model.add(Dense(output_layer_size, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

        if self.model_arch == 'bi-lstm':
            # Model 3: Bi-LSTM
            model = Sequential()
            model.add(Bidirectional(LSTM(self.batches, return_sequences=True), input_shape=(WINDOW_SIZE, 1)))
            model.add(TimeDistributed(Dense(output_layer_size, activation='sigmoid')))
            model.add(Dense(output_layer_size, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='Adam')

        if self.model_arch == 'hybrid-lstm':
            # Model 4: Hybrid-LSTM
            model = Sequential()
            model.add(Conv1D(filters=3, kernel_size=5, padding='valid', activation='relu'))
            model.add(MaxPooling1D())
            model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1), return_sequences=True))
            model.add(TimeDistributed(Dense(output_layer_size, activation='sigmoid')))
            model.add(Dense(output_layer_size, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

        self.model = model

        if self.save_model:
            self.save_structure()

    def train(self):
        # train the model
        for i in range(self.songs_epochs):
            print(f"####################################################################################################")
            print(f"################################### Songs epoch no.{i + 1} / {NUM_OF_EPOCHS} #####################################")
            print(f"####################################################################################################")

            shuffled_songs = list(zip(self.all_songs_input_windows, self.all_songs_target_windows))
            random.shuffle(shuffled_songs)
            for input_data, target_data in shuffled_songs:
                print(len(target_data[0]))
                fixed_input = input_data[:-1]

                if self.model_arch == 'bi-lstm':
                    bi_lstm_target = to_categorical(input_data[1:], self.notes_hash.get_size())
                    self.model.fit(fixed_input, bi_lstm_target, batch_size=self.batches, epochs=self.epochs)
                else:
                    self.model.fit(fixed_input, target_data, batch_size=self.batches, epochs=self.epochs)

            if self.save_weights:
                self.save_model_weights()

    def generate_MIDI(self, initial_sample: list, length):
        length = length - WINDOW_SIZE
        current_window = initial_sample
        current_window_list = list(current_window)
        total_song = list(initial_sample)
        # print(total_song)
        for i in range(length):
            current_window = np.array([current_window_list])
            current_window = np.reshape(current_window, (current_window.shape[0], current_window.shape[1], 1))

            y = self.model.predict_classes(current_window)
            # print(y.shape)
            # print(y[0][-1])
            if self.model_arch == 'bi-lstm':
                current_window_list += [int(y[0][-1])]
                current_window_list.pop(0)
                total_song += [int(y[0][-1])]
            else:
                current_window_list += [int(y)]
                current_window_list.pop(0)
                total_song += [int(y)]

        return total_song

    def write_midi_file_from_generated(self, generated, midi_file_name="result.mid", start_index=0, max_generated=1000):
        notes_list = [self.notes_hash.reversed_notes_dict[note_key] for note_key in generated]

        array_piano_roll = np.zeros((128, max_generated + 1), dtype=np.int16)
        for index, note in enumerate(notes_list[start_index:max_generated]):
            if note == 'e':
                pass
            else:
                splitted_note = list(map(str.strip, note.strip('][').replace('"', '').split(' ')))
                for pitch in splitted_note:
                    if pitch == '':
                        continue
                    array_piano_roll[int(pitch), index] = VELOCITY_CONST
        array_piano_roll = np.reshape(array_piano_roll, (array_piano_roll.shape[0], array_piano_roll.shape[1]))
        pretty_midi_obj = piano_roll_to_pretty_midi(array_piano_roll, fs=SAMPLING_FREQ, program=0)
        pretty_midi_obj.write(midi_file_name)

    def save_structure(self, saved_name="model.json"):
        # Create model's JSON
        model_json = self.model.to_json()
        with open(saved_name, "w") as json_file:
            json_file.write(model_json)

    def save_model_weights(self, saved_name='model_weights.h5'):
        self.model.save_weights(saved_name)

    def save_notes_hash(self, saved_name="Notes_hash.pickle"):
        # Save notes_hash for future use after model train
        file_to_store = open(saved_name, "wb")
        pickle.dump(self.notes_hash, file_to_store)
        file_to_store.close()

    def load_model_struct(self, path="model.json"):
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        print("Loaded Model's struct from disk...")

    def load_model_weights(self, path="model_weights.h5"):
        # load weights into new model
        self.model.load_weights(path)
        print("Loaded weights from disk...")

    def load_model_notes_hash(self, path="Notes_hash.pickle"):
        # Load notes_hash
        file_to_read = open(path, "rb")
        loaded_object = pickle.load(file_to_read)
        file_to_read.close()
        self.notes_hash = loaded_object
        print("Loaded Notes_hash from disk...")

    def load_all_model(self, struct_path="model.json", weights_path="model_weights.h5", hash_path="Notes_hash.pickle"):
        self.load_model_struct(struct_path)
        self.load_model_weights(weights_path)
        self.load_model_notes_hash(hash_path)
        print("---Loading completed---")

def generate_and_write_examples(model, path, files):
    real_notes_list, input_windows, target_windows = midi_preprocess(path=path + files[0], notes_hash=model.notes_hash,
                                                                     print_info=True, separate_midi_file=False)
    model.load_all_model()

    generated = model.generate_MIDI(list(input_windows[WINDOW_SIZE].flatten()),
                                    length=GENERATED_SONG_DURATION * SAMPLING_FREQ)
    print(generated)
    model.write_midi_file_from_generated(generated, midi_file_name="Generated_from_sample.mid",
                                         start_index=0, max_generated=GENERATED_SONG_DURATION * SAMPLING_FREQ)
    print(real_notes_list)
    model.write_midi_file_from_generated(real_notes_list, midi_file_name="Generated_real_song.mid",
                                         start_index=0, max_generated=GENERATED_SONG_DURATION * SAMPLING_FREQ)

    generated = model.generate_MIDI([0] * (WINDOW_SIZE - 1) + [2], length=GENERATED_SONG_DURATION * SAMPLING_FREQ)
    print(generated)
    model.write_midi_file_from_generated(generated, midi_file_name="Generated_from_one_note.mid",
                                         start_index=0, max_generated=GENERATED_SONG_DURATION * SAMPLING_FREQ)

def main():
    path = 'classic_piano/'
    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    print(files)
    model = ModelTrainer(files=files, path=path, model_arch='stacked-lstm', song_epochs=NUM_OF_EPOCHS, epochs=1, batches=256,
                         save_weights=True, save_model=True, save_hash=True)
    model.preprocess_files()
    model.create_model()
    model.train()

    ################################ for debugging ####################################
    # real_notes_list, input_windows, target_windows = midi_preprocess(path=path + files[0], notes_hash=model.notes_hash,
    #                                                                  print_info=True, separate_midi_file=False)
    ########################## Uncomment to generate examples #############################
    # generate_and_write_examples(model=model, path=path, files=files)

    # # draw_compare_graph(real_input=real_notes_list, predicted_input=pred_notes_list, time=midi_length)
    #compare_real_pred_notes(real_notes_list[:GENERATED_SONG_DURATION * SAMPLING_FREQ], model_generated[:GENERATED_SONG_DURATION * SAMPLING_FREQ])

if __name__ == '__main__':
    main()
