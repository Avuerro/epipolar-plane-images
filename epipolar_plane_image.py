import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import os


class EpipolarPlaneImage(object):
    """
        A class that takes as input a video and constructs a Epipolar plane image
        
    """

    def __init__(self, front, side, top, output_dir, filename):
        self.front = self._scale_image(front)
        self.side =  self._scale_image(side)
        self.top =  self._scale_image(top)
        self.output_dir = output_dir 
        self.filename = filename

    def _scale_image(self,image):
        if image.max() > 100: #use 100 to be sure 
            return image/255.
        return  image

    @classmethod
    def load_from_video(cls, video, time_start, time_end, volume_height, output_dir, filename):
        # pdb.set_trace()
        capture = cv2.VideoCapture(video)
        images = cls._get_frames(capture, time_start, time_end)
        front,side, top = cls._extract_sides(images, volume_height, output_dir, filename)
        volume_image = cls(front, side, top, output_dir, filename) 
        return volume_image

    @classmethod
    def load_from_array(cls, images, volume_height, output_dir, filename):
        images = np.asarray(images)
        volume_height = volume_height
        front,top,side = cls._extract_sides(images, volume_height, output_dir, filename) # TODO should contain filename as well
        volume_image = cls(front,side,top,output_dir, filename)
        print(f"front shape {front.shape} and top shape {top.shape} and side shape {side.shape}")
        return volume_image

    @staticmethod
    def _get_frames(capture, time_start, time_end):
        total_nr_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
        start = time_start * frame_rate
        end = time_end * frame_rate
        images = []
        for i in range(start,end):
            capture.set(1,i)
            ret, frame = capture.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(img)
        capture.release()
        cv2.destroyAllWindows()
        images = np.asarray(images)
        return images
    
    @staticmethod
    def _extract_sides(images, volume_height, output_dir, filename):
        images = np.asarray(images)
        volume_height = volume_height
        front = images[0][volume_height:,:]
        top = images[:,volume_height: volume_height+1,].squeeze()
        side =  images[:,:,-20].squeeze().transpose(1,0,2) 
        side = side[volume_height:,:]
        print(f"front shape {front.shape} and top shape {top.shape} and side shape {side.shape}")

        if not os.path.exists(f'./{output_dir}'):
            os.mkdir(f'./{output_dir}')
        ## save figures for later inspection
        plt.imsave(fname=f'./{output_dir}/{filename}_epi_front.png', arr=front, format='png') #custom filename?
        plt.imsave(fname=f'./{output_dir}/{filename}_epi_side.png',  arr=side, format='png') #custom filename?
        plt.imsave(fname=f'./{output_dir}/{filename}_epi_top.png', arr=top, format='png') #custom_filename?

        return front,side,top
    
    def construct_volume(self):

        epi = VolumeImage(self.front, self.side, self.top, self.output_dir ,self.filename)

        epi.construct_volume()



## builder pattern
class VolumeImage(object):
    
    """
        A class capable of constructing volume images
    """

    def __init__(self, front, side, top, output_dir, filename):
        self.front = front
        self.side = side
        self.top = top
        self.H, self.W,_ = self.front.shape # width and height of the volume (rectangular prism)
        _, self.T, _ = self.side.shape #time dimension, e.g. nr of frames
        self.output_dir = output_dir
        self.filename = filename

    @classmethod
    def load_images_first(cls, front,side,top, filename):
        front = cv2.imread(front)
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        side = cv2.imread(side)
        side = cv2.cvtColor(side, cv2.COLOR_BGR2RGB)
        top =  cv2.imread(top)
        top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)
        volume_image = cls(front,side,top,filename)
        return volume_image

    def _determine_surface_shapes(self):    
        
        """        
            x, y, z = width, time(number of frames), height
            Create the following meshgrids:
            front = x,z
            side = y,z
            top = x,y
        """

        xx = np.arange(0,self.W,1)
        yy = np.arange(0,self.T,1) 
        zz = np.arange(0,self.H,1)

        #front
        XF, YF = np.meshgrid(xx,zz)

        #side
        XS, YS = np.meshgrid(yy,zz)

        #top 
        XT,YT = np.meshgrid(xx,yy)

        return (XF,YF), (XS,YS), (XT,YT)

    def construct_volume(self):
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, projection='3d')
        ## obtain surface shapes tuple containing meshgrids for X and Y
        front,side,top = self._determine_surface_shapes()
        
        ax.set_axis_off()
        ## frontal surface
        ax.plot_surface(front[0], 
            np.zeros(front[1].shape), 
            front[1], 
            facecolors=np.flipud(self.front),
            rstride=2, 
            cstride=2,
            antialiased=True, 
            shade=False)
        ##side surface
        ax.plot_surface(np.zeros(side[0].shape)+self.W,
            side[0],
            side[1],
            facecolors=np.flipud(self.side),
            rstride=2,
            cstride=2,
            antialiased=True,
            shade=False)
        ##top surface
        ax.plot_surface(top[0],
            top[1],
            np.zeros(top[0].shape)+self.H,
            facecolors=(self.top),
            rstride=2, 
            cstride=2, 
            antialiased=True, 
            shade=False)
        plt.axis('off')
        plt.savefig(f'./{self.output_dir}/{self.filename}')
        plt.show()

    