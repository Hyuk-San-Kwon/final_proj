import augly.video as vidaugs

def brightness(video_path, output_path, level) :
    augmented_video = vidaugs.brightness(video_path=video_path, output_path=output_path, level=level)
    return augmented_video

def add_noise(video_path, output_path, level) :
    augmented_video = vidaugs.add_noise(video_path=video_path, output_path=output_path, level=level)
    return augmented_video

def change_video_speed(video_path, output_path, factor) :
    augmented_video = vidaugs.change_video_speed(video_path=video_path, output_path=output_path, factor=factor)
    return augmented_video

def color_jitter(video_path, output_path, brightness_factor, contrast_factor, saturation_factor) :
    augmented_video = vidaugs.color_jitter(video_path=video_path, output_path=output_path, brightness_factor=brightness_factor, contrast_factor=contrast_factor, saturation_factor=saturation_factor)
    return augmented_video

def encoding_quality(video_path, output_path, quality) :
    augmented_video = vidaugs.encoding_quality(video_path=video_path, output_path=output_path, quality=quality)
    return augmented_video

def grayscale(video_path, output_path) :
    augmented_video = vidaugs.grayscale(video_path=video_path, output_path=output_path)
    return augmented_video

def rotate(video_path, output_path, degrees) :
    augmented_video = vidaugs.rotate(video_path=video_path, output_path=output_path, degrees=degrees)
    return augmented_video