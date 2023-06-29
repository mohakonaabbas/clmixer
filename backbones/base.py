
import cv2
import numpy as np
import os.path as osp
import os
from copy import deepcopy
import random
import warnings
warnings.filterwarnings("ignore")#, "Corrupt EXIF data", UserWarning)

import random
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18,resnet152,resnet50
import torch.nn as nn
import json

from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import torchvision.transforms as T
from tqdm import tqdm
import functools

from typing import List,Union
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class BaseDataset(torch.utils.data.Dataset):
    """
    This class is a base dataset that implements a dataset.
    By default, the input image can be a folder on the hardisk or a tuple of array 
    The image can be passed through a non trainable backbone.
    This dataset class implement a memory

    """
    def __init__(self, 
                 url,
                 name,
                 backbone_name,
                 mode,
                 save_embedding,
                 save_folder=None):
        
        """
        url :the url of the dataset
        name: name of the dataset
        backbone : the backbone to be used
        mode: the mode of the training train or test
        buffer_on: wether we bufferize the data
        save_embedding: save the embeddings on the harddisk

        
        """
        # random seeder

        self.random_seed=31101994
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # FILES
        # Load the files in the root folder
        self.dataset_name=name
        self.url=url
        self.files,self.Y=self.load(url,rejectedClasses,mode)


        self.save_embedding=save_embedding
        self.saving_folder=save_folder
        self.mode=mode

        # BACKBONE LOADING 
        # Load the right backbone in eval mode
        self.backbone_name=backbone_name
        self.backbone
        # backbone data transform
        self.data_transform=
# Load the sam VIT Network , extract the features dans save them for further use

        sam_checkpoint = "./sam_vit/sam_vit_b_01ec64.pth"
        model_type = "vit_b"


        model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor=SamPredictor(model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=torch.device(device)
        
        self.model=model
        self.model.to(device=device)
        



        # INCREMENTAL LEARNING
        
        self.current_experiment=-1 # Before the beginning of the training procedure
        self.counts=np.unique(self.Y,return_counts=True)[1] # list(map(lambda x: len(os.listdir(os.path.join(url,x))),self.named_labels))
        print(f" Weights :  {self.counts}")
        
        self.weights=max(self.counts)/np.array(self.counts)
        
        self.datasets,self.datasets_labels=self.splitDataset() # SPlit the dataset according to an industrial scheme

        self.activated_files_subset =  self.datasets[0] # The current split
        self.activated_files_labels_subset = self.datasets_labels[0] # The current split
        self.nbrs_known_classes=(self.splits[:,0]>0).sum()
        self.max_classes=len(self.splits[:,0])
        
        self.output_sam_vit=(256,64,64)

        # Memory manager
        self.activated_files_subset_memory=[]
        self.activated_files_labels_subset_memory=[]

        # Current batch indexes or paths Batch indexes
        self.current_batch_paths=[]
        # Buffer to speed up the training
        self.buffer_x={}

    def step(self):
        self.current_experiment+=1
        self.activated_files_subset=self.datasets[self.current_experiment]
        self.activated_files_labels_subset=self.datasets_labels[self.current_experiment]

        # Add memory
        self.activated_files_subset+=self.activated_files_subset_memory
        self.activated_files_labels_subset+=self.activated_files_labels_subset_memory

    def splitDataset(self):
        """
        Split dataset according randomly to simulate an industrial continuous process
        """
        # Create the splits matrix
        # n1 | n2 | ... |nn
        # .. | .. |.....|
        # f1 | f2 | ... |fn
        #  Lines sum to counts

        # self.n_splits=n_splits
        splits=np.random.rand(self.counts.shape[0],self.n_splits)
        splits=splits/np.sum(splits,axis=1).reshape(-1,1)
        splits=splits*self.counts.reshape(-1,1)
        self.splits=splits.astype(np.int16)
        print(self.splits)


        labels=np.unique(self.Y)

        # Create the experiments datasets
        

        datasets=[ [] for i in range(len(labels))]
        datasets_labels=[ [] for i in range(len(labels))]
        # Indexer for data split in the list
        index=np.cumsum(self.splits,axis=1)
        index=list(map(lambda x: [0]+x,index.tolist()))

        for i,lbl in enumerate(labels):
            x_lbl=(np.array(self.files)[self.Y==lbl]).tolist()
            
            for j in range(self.n_splits):
                s,e=index[i][j],index[i][j+1]
                data=x_lbl[s:e]
                d_label=np.ones(len(data))*lbl
                datasets[i].append(data)
                datasets_labels[i].append(d_label)
        

        datasets=list(zip(*datasets))
        datasets_labels=list(zip(*datasets_labels))


        # self.datasets=datasets # list(map(lambda x:np.concatenate(x),datasets))
        # self.datasets_labels=datasets_labels #  list(map(lambda x:np.concatenate(x),datasets_labels))

        self.datasets=list(map(lambda z: functools.reduce(lambda x,y :x+y,z),datasets))
        self.datasets_labels=list(map(lambda x:np.concatenate(x).tolist(),datasets_labels))



        return self.datasets,self.datasets_labels

    def load(self,
             url : str,
             rejectedClasses: List[str],
             mode : str):
        """
        Load on the disk the files
        It assumes that each folder represent a class.
    
        """
        self.dataset_image_path = os.path.join(url,"data")
        # Get the labels names : Title of the folder
        self.named_labels=os.listdir(self.dataset_image_path)
        self.labels=np.arange(len(self.named_labels),dtype=float)
        # Create an ordered dict mapping int to 
        self.label_dict=dict(zip(self.named_labels,self.labels))

        self.rejected_classes=rejectedClasses
        
        self.n_classes=len(self.labels)

        # Load the files
        # Check if the train and test files already exists
        pre_split=os.path.exists(os.path.join(url,"train.txt"))
        if pre_split: #No : Is there a predefined files in the folder of the data . Named train.txt and test.txt ?
            train_files=np.loadtxt(os.path.join(url,"train.txt"))
            test_files=np.loadtxt(os.path.join(url,"test.txt"))
        else:
            self.files,labels=self.crawlDataFolder(self.dataset_image_path,self.rejected_classes)

            # Many case can happen here
            # Is there is a test and train set in the dataset folder ?
            # Yes : Use it . No : Random shuffle the data and do a 80/20 split . Save in files named train and test.txt the split

            default_train_set=list(map(lambda x : "train/" in x,self.files))
            train_files=self.files[default_train_set]
            if len(train_files)>0: 
                # Is there is a test and train set in the dataset folder ?            
                default_test_set=list(map(lambda x : "train/" not in x,self.files))
                test_files=self.files[default_test_set]
            else:
                
                # Use the mode to generate the splits ( )
                alpha=0.8
                split_idx=int(alpha*len(self.files))
                files=random.shuffle(self.files)
                train_files=np.array(files[:split_idx])
                test_files=np.array(files[split_idx:])

            np.savetxt(os.path.join(url,"train.txt"),train_files)
            np.savetxt(os.path.join(url,"test.txt"),test_files)

        

        if mode=="train":
            files=train_files.tolist()
        elif mode=="test":
            files=test_files.tolist()

        lbl_getter=lambda x: self.label_dict[x.replace(self.dataset_image_path,"").split('/')[0]]
        Y=list(map(lbl_getter,files))
        return files,np.array(Y)

    def __len__(self):

        return len(self.activated_files_subset)

    def __getitem__(self, idx):

        """
        1- Try to Use the buffer value if availabale
        2- Try to use the already saved embeddings if available
        3- Load the image file and use it with backbone

        """
        try:
            x=self.buffer_x[self.activated_files_subset[idx]][0]
            y=self.buffer_x[self.activated_files_subset[idx]][1]
            
        except:

            img=cv2.imread(self.activated_files_subset[idx])
            img=cv2.resize(img, (244,244), interpolation = cv2.INTER_NEAREST)
            self.current_batch_paths.append(self.activated_files_subset[idx])
            lbl=self.label_dict[self.activated_files_subset[idx].split('/')[-1*self.level]]# Get the numeric label 
            lbl=int(lbl)
            

            # processed_x=self.batch_sam_preprocess(img)
            # processed_x=processed_x.to(self.device)
            # processed_x = self.model.preprocess(processed_x)
            with torch.no_grad():
                
                self.sam_predictor.set_image(img)

                x = self.sam_predictor.features.reshape((self.output_sam_vit[0],self.output_sam_vit[-1]**2)) #.image_encoder(processed_x)
                
                self.sam_predictor.reset_image()

                

            # x= self.X[idx,:]
            # y=lbl

            y=torch.tensor(lbl,dtype=torch.long)
            self.buffer_x[self.activated_files_subset[idx]]=(x,y)
        return x, y

    def crawlDataFolder(self,
                        entryDirectory : str,
                        reject : List[str]=[]):
        """
        Look in all subfolders for any files with the requested extension
        """
        foundFilesPaths=[]
        foundFilesLabels=[]
        searchedExtension =['jpg','png','bmp','jpeg']

        for root, dir_names, file_names in os.walk(entryDirectory):
            
            for f in file_names:
                rejections=list(map(lambda x: x in root,reject))
                if any(rejections):
                    continue

                fname = os.path.join(root, f)
                extension=os.path.splitext(fname)

                if extension.lower() in searchedExtension:
                    foundFilesPaths.append(fname)

                    label=fname.replace(entryDirectory,"").split("/")[0]
                    foundFilesLabels.append(label)


        return foundFilesPaths,foundFilesLabels
    
    def generateEmbeddings(self,model,files, savingFolder,level=2,bs=1024):
        """
        This function generates embeddings for the whole dataset at once and saves it in a folder
        """

        # Save in a folder
        if not os.path.exists(savingFolder):
            # Create the folders
            os.makedirs(savingFolder)
        if not os.path.exists(os.path.join(savingFolder,"data")):
            os.makedirs(os.path.join(savingFolder,"data"))
        if not os.path.exists(os.path.join(savingFolder,"labels")):
            os.makedirs(os.path.join(savingFolder,"labels"))
        
        X= np.zeros((len(files),256,64,64),dtype=np.float32)
        Y= np.zeros((len(files)),dtype=np.uint8)


        self.model.to(self.device)
        for i in tqdm(range(0,len(files),bs)):
            imgs=list(map(lambda x: cv2.resize(cv2.imread(x), (244,244), interpolation = cv2.INTER_NEAREST),files[i:i+bs]))
            lbs=list(map(lambda x:self.label_dict[x.split('/')[-1*level]],files[i:i+bs])) # Get the numeric label 
            x_=np.array(imgs)
            processed_x=self.batch_sam_preprocess(x_)
            
            # x_=x_.transpose(0,3,1,2)/255.0
            # x_=(x_- self.mean)/self.std
            # x_=torch.tensor(x_,dtype=torch.float32).to(self.device)
            # x_=x_.transpose(0,3,1,2)
            
            processed_x=processed_x.to(self.device)
            processed_x = self.model.preprocess(processed_x)
            with torch.no_grad():
                x_embeddings = self.model.image_encoder(processed_x)
                # x_embeddings=model(x_) # bsx512
                embeddings=x_embeddings.cpu().numpy()
                X[i:i+bs,:]=embeddings
                Y[i:i+bs]=np.array(lbs)

        
        

        np.save(os.path.join(savingFolder,"data/X.npy"),X)
        np.save(os.path.join(savingFolder,"labels/y.npy"),Y)
        return X,Y

class simpleDataset(torch.utils.data.Dataset):
    def __init__(self, X:list[str],
                 y:list[int],
                 sam_predictor: SamPredictor):
        self.X=X
        self.y=y
        self.sam_predictor=sam_predictor
        self.output_sam_vit=(256,64,64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        img=cv2.imread(self.X[idx])
        img=cv2.resize(img, (244,244), interpolation = cv2.INTER_NEAREST)
        
        lbl=self.y[idx] 
        

        # processed_x=self.batch_sam_preprocess(img)
        # processed_x=processed_x.to(self.device)
        # processed_x = self.model.preprocess(processed_x)
        with torch.no_grad():
            
            self.sam_predictor.set_image(img)

            x = self.sam_predictor.features.reshape((self.output_sam_vit[0],self.output_sam_vit[-1]**2)) #.image_encoder(processed_x)
            
            self.sam_predictor.reset_image()

        # x= self.X[idx,:]
        # y=lbl

        y=torch.tensor(int(lbl),dtype=torch.long)
        return x, y

if __name__=="__main__":

    n_splits=5


    path="/home/mohamedphd/Documents/phd/classical/data/images"
    
    load_dict={"nouvelop":nouvelOpDataset,"kth":kth,'easydataset':eaydataset,"mvtec":mvtec,"kaggleTextureSynthetic":kaggleSynthetic,"MagneticTile":tiles,"severstal":severstal}
    datasets=list(load_dict.values())
    names=list(load_dict.keys())
    
    paths=["/home/mohamedphd/Documents/phd/classical/data/images",
           "/home/mohamedphd/Documents/phd/Acquisition Datasets Internet/kth",
           "/home/mohamedphd/Documents/phd/Acquisition Datasets Internet/createdEasyDataset",
           "/home/mohamedphd/Documents/phd/Acquisition Datasets Internet/mvtec",
           "/home/mohamedphd/Documents/phd/Acquisition Datasets Internet/dagm_kaggleupload/DAGM_KaggleUpload",
           "/home/mohamedphd/Documents/phd/Acquisition Datasets Internet/Magnetic-tile-defect-datasets.-master",
           "/home/mohamedphd/Documents/phd/Acquisition Datasets Internet/severstal-steel"]
    rejectedClasses=[[],["no_damage","NAGEL"],[],[],[],[],[]]
    #path="/home/mohamedphd/Documents/phd/embeddings/nouvel/"
    mode="raw"


    for key,path,rejectedClass in zip(names,paths,rejectedClasses):
        if not key == "mvtec":continue
        print(f"\nDataset : {key} -- Mode : {mode}")
        embeddingPath=os.path.join("/home/mohamedphd/Documents/phd/embeddings",key)
        path=embeddingPath if mode=="embeddings" else path
        dataset=load_dict[key](path,rejectedClasses=rejectedClass,mode=mode)
        for input,lbl in dataset:
            print(input.shape, lbl)
        dataset.step()

