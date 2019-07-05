import os
import time
import webbrowser
import _thread
import shutil

'''open tensorboard'''
def open_tensorboard(dir):

    def execute_se():
        os.popen('tensorboard --logdir='+dir)


    def open_web():
        time.sleep(4)
        webbrowser.open_new('http://localhost:6006')
        #webbrowser.open_new('http://127.0.0.1:6006')
    #empty file
    try:
        shutil.rmtree(dir)
    except:
        'do nothing'

    os.mkdir(dir)
    #
    _thread.start_new_thread(execute_se, ())
    _thread.start_new_thread(open_web, ())

if __name__ == '__main__':

    import sys
    rootpath = sys.path[0]
    dirs = rootpath[:-6] + '/FSnum_3layersConnect_50/'
    print(dirs)
    os.popen('tensorboard --logdir=' + dirs)
    webbrowser.open_new('http://localhost:6006')



    #tensorboard --logdir=C:/Users/abc123/Desktop/IDS/IDS/logs/FSnum_3layersConnect_50/
    #tensorboard --logdir=C:/Users/abc123/Desktop/IDS/IDS/logs/FSnum_50/

    #C:/Users/abc123/Desktop/IDS/IDS/logs/FSnum_50/
