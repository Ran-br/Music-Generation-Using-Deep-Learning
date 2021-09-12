# MIDI_music_generation

Music has been for thousands of years a source of great pleasure for the human kind, but for most of human history, it has been created only by humans, in various ways. For decades, researchers had been trying to create a machine which generates music without the creativity of the human mind. Recent years have brought great prosperity to computer generation methods in many fields, probably most prominent of them is the field of image generation. Using these new techniques, researchers have been able generate amazingly realistic images, especially of human faces. Video and voice generation have seen great improvements too, as can be seen in the greatly hyped ‘deep fake’ industry which has emerged based upon them. Images are very fast and easy to grasp by us humans, making the result of image generation easier to grasp and be amazed by, while videos and voice generation have enormous demand. Music generation exists and had major developments as well, but has seen less publicity, in our opinion, due to the lack of public interest in it and the fact that it requires more attention in order to be perceived. 

In this project, we take advantage of a well known model called LSTM and use several different architectures based on it, in order to tackle the problem of generating music. At first, as the base part, we will review a simple model architecture that consists of a single LSTM layer and a single output layer. After discussing the results and conclusion of this part, we will review both Bi-LSTM, Hybrid LSTM and Stacked LSTM, some of which will show great improvements over the base model, and some won’t. We will then offer some more research clues which we suspect could lead to even better results in the future.
