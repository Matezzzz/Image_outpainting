import glob
import os
import multiprocessing

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


dloc = os.getenv("IMAGE_OUTPAINTING_DATASET_LOCATION")

def scan_location(loc):
    all_imgs = glob.glob(f"{loc}/*/*.jpg")
    for i, img in enumerate(all_imgs):
        if i % 10000 == 0:
            print (f"{loc} at {i} out of {len(all_imgs)}")
        try:
            img_file = tf.io.read_file(img)
            tf.image.decode_image(img_file, 3, expand_animations=False)
        except:
            print (f"Loading image '{img}' failed")
            with open("dataset_errors", "a") as f:
                print (f"Loading image '{img}' failed", file=f)

#268 800

ps = []
for location in sorted(glob.glob(f"{dloc}/*")):
    print (f"Starting: {location}")
    p = multiprocessing.Process(target=scan_location, args=(location,))
    p.start()
    ps.append(p)
    while True:
        a = input("Another process (y/l)?")
        if a == 'y':
            break
        if a == 'l':
            c = sum(x.is_alive() for x in ps)
            print (f"{c} processes running, {len(ps)-c} finished")
        else:
            print ("Wrong input, try again")

for p in ps:
    p.join()
