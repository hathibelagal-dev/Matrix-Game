from diffusers.utils import export_to_video

def process_video(input_video, output_video):
    fps = 12
    frame_count = len(input_video)
    
    out_video = []
    frame_idx = 0
    for frame in input_video:
        out_video.append(frame / 255)
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}", end="\r")
    export_to_video(out_video, output_video, fps=fps)
    print("\nProcessing complete!")