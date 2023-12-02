import os
import random
import function

PATH = os.getcwd()

def augmentation(type_aug, folder_path, output_vid_path) :
    index_list = []
    
    for i in range(20) :
        index = random.randint(1, 20)
        while index in index_list:
            index = random.randint(1, 20)
    
        index_list.append(index)
        input_video_path = PATH + folder_path + str(index) + ".mp4"
        output_path = PATH + output_vid_path + str(type_aug) + "_" + str(index) + ".mp4"
        
        if type_aug == 1 :
            function.brightness(video_path=input_video_path, output_path=output_path, level=random.uniform(-0.2,0.5))
        elif type_aug == 2 :
            function.add_noise(video_path=input_video_path, output_path=output_path, level=random.randint(0,100))
        elif type_aug == 3 :
            function.change_video_speed(video_path=input_video_path, output_path=output_path, factor=random.uniform(0.5,2))
        elif type_aug == 4 :
            function.color_jitter(video_path=input_video_path, output_path=output_path, brightness_factor=0, contrast_factor=random.uniform(1,5), saturation_factor=random.uniform(0,3))
        elif type_aug == 5 :
            function.encoding_quality(video_path=input_video_path, output_path=output_path, quality=random.randint(0,51))
        elif type_aug == 6 :
            function.grayscale(video_path=input_video_path, output_path=output_path)
        elif type_aug == 7 :
            function.rotate(video_path=input_video_path, output_path=output_path, degrees=random.uniform(-10,10))
        else :
            print("NO SUCH TYPE OF AUGMENTATION!!!")
