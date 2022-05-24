# RSE_post_process
RSE Network add post-process, for music transcription

# Usage:
Need to download validation data and model data

* model(783.8M)
  * https://drive.google.com/file/d/1k_YGErqs2YfA-wkypkYhicBO5bQpOf9c/view?usp=sharing

* validation npy(11.8G)
  * https://drive.google.com/file/d/1K5H_FLfMlTNFFjoag7_SvWXjkDLya1kT/view?usp=sharing

# Execute Command

* If wav file is not clear audio, then use this program to run denoising process and get tempo
   * Input: wav file
   * Output: denoised wav file
   * `python execute_all_preprocessing.py --input <filename>`


* Turn wav into npy
    * Input: wav file
    * Output: npy file
    * `python make_dataset.py --input <filename>_reduction`


* Produce visualisation.npy
    * Input: npy file
    * output: npy file for visualization
    * `python visualiser.py --input <filename>_reduction` or `python visualiser.py --input <filename>`

* Produce visualisation.npy
    * Input: npy file
    * output: midi file (final result)
    * `python show_visualisation.py --input <filename>_reduction_visualisation` or `python show_visualisation.py --input <filename>_visualisation`

