import subprocess
import os
import time
import sys
import petname
import shipname
import re
from utils import timer
import go

GAMES_BUCKET = "gs://jacksona-mugo/games/"
MODEL_NUM_REGEX = "\d{6}"
GAME_DIRECTORY = "./data/selfplay/"
MODEL_DIRECTORY = "./saved_models"
TRAINING_DIRECTORY = "data/training_chunks"

def bootstrap():
    import dual_net
    n = dual_network


def push_model(model_num, name):
    for f in os.listdir('saved_models'):
        if f.startswith("{0:06d}".format(model_num)):
            arg =  'gsutil cp saved_models/%s gs://jacksona-mugo/models/%06d-%s.%s' % (
                    f, model_num, name, f.split('.')[1])
            subprocess.call(arg.split())

def dir_model_num(dir_path):
    'Returns the model number of a directory, if present, e.g. 000010-neat-model => 10'
    if not re.match(MODEL_NUM_REGEX, os.path.basename(dir_path)):
        return None
    return int(re.match(MODEL_NUM_REGEX, os.path.basename(dir_path)).group())


def smart_rsync(from_model_num):
    from_model_num = 0 if from_model_num < 0 else from_model_num
    seen_dirs = subprocess.check_output(('gsutil ls -d %s*' % GAMES_BUCKET).split()).split()
    seen_dirs = list(map(lambda d: d.decode('UTF-8').strip('/'), seen_dirs))
    model_dirs = [d for d in seen_dirs
            if dir_model_num(d) is not None and dir_model_num(d) >= from_model_num ]
    print("model_dirs:", model_dirs)
    for d in model_dirs:
        basename = os.path.basename(d)
        if not os.path.isdir(os.path.join(GAME_DIRECTORY, basename)):
            os.mkdir(os.path.join(GAME_DIRECTORY, basename))
        subprocess.call('gsutil -m rsync -r -c {0}{2} {1}{2}'.format(
                    GAMES_BUCKET, GAME_DIRECTORY, basename).split(),
                    stderr=open('.rsync_log', 'ab'))

def find_largest_modelnum(directory):
    return max([dir_model_num(f) or 0 for f in os.listdir(directory) ])

def bg_chunk_runner():
    # Check how many chunks there are.  Run rsync.  If there are no
    # chunks, run gather, else wait a bit, and repeat.
    while True:
        while True:
            num_chunks = sum([1 if p.endswith('.gz') else 0 for p in os.listdir('data/training_chunks')])
            print('found ', num_chunks, ' num_chunks')
            maxnum = find_largest_modelnum(MODEL_DIRECTORY)
            print ("Oldest model: ", maxnum)
            with timer("=== Rsync new games"):
                smart_rsync(maxnum-6)
            if num_chunks == 0:
                break
            time.sleep(30)

        # Create training chunks
        failball = subprocess.call( ('python main.py gather').split())
        if failball:
            print(failball)
            sys.exit(1)

if __name__ == '__main__':
    model_num = find_largest_modelnum(MODEL_DIRECTORY) or 0
    train_cmd = 'python main.py train --load-file saved_models/{0:06d} -s saved_models/{1:06d}' + \
                    ' --logdir logs/rl_loop_9 data/training_chunks/'

    bg_chunk_proc = multiprocessing.Process(target=bg_chunk_runner, name="bg_chunk_runner")
    bg_chunk_proc.start()
    while True:
        print(" ====== Model %d ======" % (model_num))

        print("Waiting for training chunks...")
        while True:
            num_chunks = sum([1 if fname.endswith('.gz') else 0 for fname in os.listdir(TRAINING_DIRECTORY)])
            if num_chunks != 0:
                break
            time.sleep(10)

        # Take a training step.
        bigarg = train_cmd.format(model_num, model_num+1)
        print("RUNNING '%s'" % bigarg)
        failball = subprocess.call(bigarg.split())
        if failball:
            print(failball)
            sys.exit(1)

        # Wipe the training chunks.
        for p in os.listdir('data/training_chunks'):
            if p.endswith('.gz'):
                os.remove(os.path.join('data/training_chunks/', p))

        # For now, always promote our new model.
        if go.N == 19:
            new_name = shipname.generate()
        else:
            new_name = petname.generate()
        print("A new champion! ", new_name)
        print("Pushing %06d-%s to saved_models" % (model_num, new_name))
        push_model(model_num, new_name)
        subprocess.call('./update-acls')
        model_num+=1