Current Progress
 GitHub - https://github.com/modidiviyansh/Voic-lone
We have worked on the front end and the backend during the summer vacations. 
1.     Frontend: -
Our front end is based on HTML, CSS and vanilla js. The website consists of three sections, namely:
Upload - allows the user to upload their files
Text Area - the text which has to be spoken by the cloned voice
Download - download the final audio file
The frontend UI’s concept is inspired by glass morphism.
Screenshot : 

![image](https://user-images.githubusercontent.com/71758538/189226935-5bf69f7f-279f-486e-8f7f-26d00923d989.png)




2.     Voice cloning model (Backend logic): 
Our project’s backend is an amalgamation of various TTS models, which are mentioned below:
 SC-GlowTTS: an efficient zero-shot multi-speaker text-to-speech model that improves similarity for speakers unseen during training.
VITS (used by SC-Glow TTS): a speaker-conditional architecture that explores a flow-based decoder that works in a zero-shot scenario.
Dataset
The original model was trained on VCTK and LibriTTS for English (multispeaker datasets), and later transfer learning was applied on it to make the training process faster.
Training
Our model is a transfer learning version of Glow TTS and VITS, they both combine to make an end to end system (voice to voice) that can learn a voice’s characteristics using a sample of around 5 seconds
Screenshots
.


To be done
Integrating frontend and backend
Adding animations to frontend
Deploying the model
