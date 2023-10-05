# Lip-Syncing using WavtoLip and post processing the frames using Real-ESRGAN algorithm to chieve High-Fidelity Videos

The repository combines two advanced algorithms:

- Wav2Lip - This is used for achieving accurate lip-syncing in videos, ensuring the movement of the lips matches the audio.
- Real-ESRGAN - This enhances the video resolution, making it visually sharper and clearer.

Together, these algorithms produce lip-synced videos that are not only precise in lip movements but also of high visual quality.

Here due to less computational resources i had to use weights of pre-trained models 

## Lip-Sync Training Guide

### Overview
There are two critical stages in the training process:
1. **Expert Lip-Sync Discriminator Training**
2. **Wav2Lip Model Training**

---

### 1. Expert Lip-Sync Discriminator Training
If you prefer to skip this step, pre-trained weights are available for download. If you're going to train it yourself:

```
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints>
```

---

### 2. Wav2Lip Model Training
There are two training options available:

**a. Without the Visual Quality Discriminator** (Takes approximately <1 day)
```
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```

**b. With the Visual Quality Discriminator** (Takes approximately ~2 days)
For this option, use `hq_wav2lip_train.py`. The arguments remain similar for both files. You can also resume training if needed. For more detailed instructions, run:
```
python wav2lip_train.py --help
```

For less common hyper-parameters, refer to the end of the `hparams.py` file.

---

### Training on Datasets Other Than LRS2
If you're considering training on datasets other than LRS2, here are a few things to note:

- Achieving good results by training or fine-tuning on a single speaker for just a few minutes can be challenging. This is an unresolved research problem.
- Before training Wav2Lip, ensure that the expert discriminator is trained on your dataset.
- Datasets sourced from the internet often require sync-correction.
- Be aware of your dataset's FPS (Frames Per Second). FPS changes might necessitate significant code alterations.
- Ideally:
  - The expert discriminator's evaluation loss should be around ~0.25.
  - The Wav2Lip's evaluation sync loss should be around ~0.2 for optimal outcomes.

When reporting any issues related to this, kindly confirm your awareness of the points mentioned above.

---

### Commercially Usable HD Model
We offer an HD model trained on a dataset suitable for commercial use. This model produces faces with dimensions of 192 x 288.

---

## Lip sync Infrence guide

1.  Clone this repository and install requirements using following command (Make sure, Python and CUDA are already installed):

    ```
    git clone https://github.com/vavinash992/Lip-Synx.git
    cd Wav2Lip-HD
    pip install -r requirements.txt
    cd Real-ESRGAN
    pip install -r requirements.txt
    !pip install basicsr
    !pip install facexlib
    !pip install gfpgan
    !pip install -r requirements.txt
    !python setup.py develop
    ```

2. Downloading weights

| Model        | Directory           | Download Link  |
| :------------- |:-------------| :-----:|
| Wav2Lip           | [checkpoints/](https://github.com/saifhassan/Wav2Lip-HD/tree/main/checkpoints)   | [Link](https://drive.google.com/drive/folders/1tB_uz-TYMePRMZzrDMdShWUZZ0JK3SIZ?usp=sharing) |
| ESRGAN            | [experiments/001_ESRGAN_x4_f64b23_custom16k_500k_B16G1_wandb/models/](https://github.com/saifhassan/Wav2Lip-HD/tree/main/experiments/001_ESRGAN_x4_f64b23_custom16k_500k_B16G1_wandb/models) | [Link](https://drive.google.com/file/d/1Al8lEpnx2K-kDX7zL2DBcAuDnSKXACPb/view?usp=sharing) |
| Face_Detection    | [face_detection/detection/sfd/](https://github.com/saifhassan/Wav2Lip-HD/tree/main/face_detection/detection/sfd) | [Link](https://drive.google.com/file/d/1uNLYCPFFmO-og3WSHyFytJQLLYOwH5uY/view?usp=sharing) |
| Real-ESRGAN       | Real-ESRGAN/gfpgan/weights/   | [Link](https://drive.google.com/drive/folders/1BLx6aMpHgFt41fJ27_cRmT8bt53kVAYG?usp=sharing) |
| Real-ESRGAN       | Real-ESRGAN/weights/          | [Link](https://drive.google.com/file/d/1qNIf8cJl_dQo3ivelPJVWFkApyEAGnLi/view?usp=sharing) |

3. Put input video to `input_videos` directory and input audio to `input_audios` directory.

4. Set the following env variables

    ```
    export filename= trimmed_audio    #(just video file name without extension)
    export input_video=input_videos
    export input_audio=input_audios/ai.wav    #(audio filename with extension)
    export frames_wav2lip=frames_wav2lip
    export frames_hd=frames_hd
    export output_videos_wav2lip=output_videos_wav2lip
    export output_videos_hd=output_videos_hd
    export back_dir=..

    ```

5. Run the following commands   
    
    ```
    python3 inference.py --checkpoint_path "checkpoints/wav2lip_gan.pth" --segmentation_path "checkpoints/face_segmentation.pth" --sr_path "checkpoints/esrgan_yunying.pth" --face ${input_video}/${filename}.mp4 --audio ${input_audio} --save_frames --gt_path "data/gt" --pred_path "data/lq" --no_sr --no_segmentation --outfile ${output_videos_wav2lip}/${filename}.mp4
    python video2frames.py --input_video ${output_videos_wav2lip}/${filename}.mp4 --frames_path ${frames_wav2lip}/${filename}
    cd Real-ESRGAN
    python inference_realesrgan.py -n RealESRGAN_x4plus -i ${back_dir}/${frames_wav2lip}/${filename} --output ${back_dir}/${frames_hd}/${filename} --outscale 3.5 --face_enhance

    ```



