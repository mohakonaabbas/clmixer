
import cv2
import numpy as np
import os.path as osp
import os
from pathlib import Path
import random
import warnings
warnings.filterwarnings("ignore")#, "Corrupt EXIF data", UserWarning)

import random
import torch
import torch.nn as nn

import torchvision.transforms as T
from tqdm import tqdm
import functools
import PIL
from typing import List,Union
from .backbones import Dinov2,Resnet

BACKBONES={'dinov2':Dinov2,'resnet':Resnet}

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
                 url : str,
                 name: str,
                 backbone_name : str,
                 mode : str,
                 save_embedding : bool,
                 n_splits : int):
        
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
        torch.random.manual_seed(self.random_seed)

        # FILES
        # Load the files in the root folder
        self.dataset_name=name
        self.url=url
        self.files,self.Y=self.load(url,[],mode)


        self.save_embedding=save_embedding
        self.mode=mode

        self.default_embedding_location=f"embeddings/{backbone_name}"

        
        # BACKBONE LOADING 
        # Load the right backbone in eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=torch.device(device)
        self.backbone_name=backbone_name

        if "resnet" in self.backbone_name:
            self.backbone=BACKBONES["resnet"](backbone_name=self.backbone_name,device=self.device)
        elif "dinov2" in self.backbone_name:
            self.backbone=BACKBONES["dinov2"](backbone_name=self.backbone_name,device=self.device)

        # Manage the folder to keep embeddings
        if save_embedding:
            self.generateEmbeddings()


        # INCREMENTAL LEARNING
        
        self.current_experiment=-1 # Before the beginning of the training procedure
        self.counts=np.unique(self.Y,return_counts=True)[1] # list(map(lambda x: len(os.listdir(os.path.join(url,x))),self.named_labels))
        print(f" Weights :  {self.counts}")
        
        self.weights=max(self.counts)/np.array(self.counts)
        self.n_splits=n_splits
        self.datasets,self.datasets_labels=self.splitDataset() # SPlit the dataset according to an industrial scheme
        self.activated_files_subset =  self.datasets[0] # The current split
        self.activated_files_labels_subset = self.datasets_labels[0] # The current split
        self.nbrs_known_classes=(self.splits[:,0]>0).sum()
        self.max_classes=len(self.splits[:,0])
        
        self.output_shape=self.backbone.output_shape

        # Memory manager
        self.activated_files_subset_memory=[]
        self.activated_files_labels_subset_memory=[]

        # Current batch indexes or paths Batch indexes
        self.current_batch_paths=[]
        # Buffer to speed up the training
        self.buffer_x={}

    def step(self):
        self.current_experiment+=1
        # We have a different behaviour based onn the mode we are in
        if self.mode == "test":
            self.activated_files_subset=self.files
            self.activated_files_labels_subset=self.Y
        else:
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
        mask= np.random.choice([True,False],size=splits.shape)
        splits=splits/np.sum(splits,axis=1).reshape(-1,1)
        splits=splits*self.counts.reshape(-1,1)
        self.splits=mask*splits.astype(np.int16)+np.roll(~mask*splits.astype(np.int16),1,axis=1)
        
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
        self.labels=np.arange(len(self.named_labels),dtype=int)
        # Create an ordered dict mapping int to 
        self.label_dict=dict(zip(self.named_labels,self.labels))
        self.label_dict_inverted=dict(zip(self.labels,self.named_labels))

        self.rejected_classes=rejectedClasses
        
        self.n_classes=len(self.labels)

        # Load the files
        # Check if the train and test files already exists
        pre_split=os.path.exists(os.path.join(url,"train.txt"))
        if pre_split: #No : Is there a predefined files in the folder of the data . Named train.txt and test.txt ?
            train_files=np.loadtxt(os.path.join(url,"train.txt"),dtype="str")
            test_files=np.loadtxt(os.path.join(url,"test.txt"),dtype="str")
        else:
            self.files,labels=self.crawlDataFolder(self.dataset_image_path,self.rejected_classes)

            # Many case can happen here
            # Is there is a test and train set in the dataset folder ?
            # Yes : Use it . No : Random shuffle the data and do a 80/20 split . Save in files named train and test.txt the split

            default_train_set=list(map(lambda x : "train/" in x.lower(),self.files))
            train_files=np.array(self.files)[default_train_set]
            
            if len(train_files)>0: 
                # Is there is a test and train set in the dataset folder ?            
                default_test_set=list(map(lambda x : "train/" not in x.lower(),self.files))
                test_files=np.array(self.files)[default_test_set]
                
            else:
                
                # Use the mode to generate the splits ( )
                alpha=0.8
                split_idx=int(alpha*len(self.files))
                random.shuffle(self.files)
                train_files=np.array(self.files[:split_idx])
                test_files=np.array(self.files[split_idx:])

            np.savetxt(os.path.join(url,"train.txt"),train_files,fmt="%s")
            np.savetxt(os.path.join(url,"test.txt"),test_files,fmt="%s")

        

        if mode=="train":
            files=train_files.tolist()
        elif mode=="test":
            files=test_files.tolist()

        lbl_getter=lambda x: self.label_dict[x.replace(self.dataset_image_path,"").split('/')[1]]
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
        path=self.activated_files_subset[idx]

        # Manage the folder to keep embeddings
        embedding_path=path.replace('data',self.default_embedding_location)
        _,extension=os.path.splitext(path)
        embedding_path=embedding_path.replace(extension,".pt")
        embedding_path_folder=os.path.dirname(embedding_path)
        if not os.path.exists(embedding_path_folder) : os.makedirs(embedding_path_folder)

        try:
            #Try to use the embeddings
            x=self.buffer_x[path][0]
            y=self.buffer_x[path][1]
            
        except:
            # Try to load the embeddings
            if os.path.exists(embedding_path):
                x=torch.load(embedding_path)
                lbl=self.Y[self.files.index(path)]
                
            else:
                img=cv2.imread(path)
                x=self.backbone.predict(img)
                #Save it in the embedding folder
                lbl=self.Y[self.files.index(path)]
                torch.save(x.detach().cpu(),embedding_path)
        
            y=torch.tensor(lbl,dtype=torch.long)

        self.current_batch_paths.append(path)
        self.buffer_x[path]=(x,y)
        return x, y

    def crawlDataFolder(self,
                        entryDirectory : str,
                        reject : List[str]=[]):
        """
        Look in all subfolders for any files with the requested extension
        """
        foundFilesPaths=[]
        foundFilesLabels=[]
        searchedExtension =['*.jpg','*.png','*.bmp','*.jpeg','*.PNG']
        
        for pattern in searchedExtension:
            names=Path(entryDirectory).rglob(pattern)
            for path in names:
                fname = str(path)
                label=fname.replace(entryDirectory,"").split("/")[1]
                foundFilesLabels.append(label)
                foundFilesPaths.append(fname)


        return foundFilesPaths,foundFilesLabels
    
    def generateEmbeddings(self):
        """
        This function generates embeddings for the whole dataset at once and saves it in a folder
        """

        for i in tqdm(range(0,len(self.files))):
            path=self.files[i]
            embedding_path=path.replace('data',self.default_embedding_location)
            _,extension=os.path.splitext(path)
            embedding_path=embedding_path.replace(extension,".pt")

            if os.path.exists(embedding_path) :
                # Do not regenerate the same file 
                continue


            embedding_path_folder=os.path.dirname(embedding_path)
            if not os.path.exists(embedding_path_folder) : 
                os.makedirs(embedding_path_folder)

            img=cv2.imread(path)
            x=self.backbone.predict(img)
            #Save it in the embedding folder
            torch.save(x.detach().cpu(),embedding_path)


class simpleDataset(torch.utils.data.Dataset):
    def __init__(self, X:list[str],
                 y:list[int],
                 predictor):
        self.X=X
        self.y=y
        self.predictor=predictor
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        """
        1- Try to Use the buffer value if availabale
        2- Try to use the already saved embeddings if available
        3- Load the image file and use it with backbone

        """
        path=self.X[idx]
        default_embedding_location=f"embeddings/{self.predictor.backbone_name}"

        # Manage the folder to keep embeddings
        embedding_path=path.replace('data',default_embedding_location)
        _,extension=os.path.splitext(path)
        embedding_path=embedding_path.replace(extension,".pt")
        
        # Try to load the embeddings
        if os.path.exists(embedding_path):
            x=torch.load(embedding_path)
            
            
        else:
            img=cv2.imread(path)
            x=self.predictor.predict(img)
            #Save it in the embedding folder
            torch.save(x.detach().cpu(),embedding_path)
        
            # y=torch.tensor(lbl,dtype=torch.long)



        lbl=self.y[idx] 

        y=torch.tensor(int(lbl),dtype=torch.long)
        return x, y

if __name__=="__main__":

    n_splits=5


    root_folder_path="/home/mohamedphd/Documents/phd/Datasets/curated/"

    datasets_names=os.listdir(root_folder_path)
    backbones=["dinov2_vits14","resnet18"]
    modes = ["test","train"]

    for backbone in backbones:
        for dataset_name in  datasets_names:
            for mode in modes:
                path=os.path.join(root_folder_path,dataset_name)
                print(f" Dataset : {dataset_name}\n Backbone : {backbone}\n mode : {mode} \n------------------------")

                dataset=BaseDataset(url=path,
                                    name=dataset_name,
                                    backbone_name=backbone,
                                    mode=mode,
                                    save_embedding=True,
                                    n_splits=n_splits)
    
                for input,lbl in dataset:
                    print(input.shape, lbl)
                    break


