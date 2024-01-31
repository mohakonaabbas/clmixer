

import numpy as np

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

from typing import List,Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import shutil


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class DummyBaseDataset(torch.utils.data.Dataset):
    """
    This class is a base dataset that implements a dataset.
    By default, the input image can be a folder on the hardisk or a tuple of array 
    The image can be passed through a non trainable backbone.
    This dataset class implement a memory

    """
    def __init__(self, 
                 url : str,
                 name: str,
                 backbone_name : str ,
                 mode : str,
                 save_embedding : bool,
                 n_splits : int,
                 split_mode : str = "cil",
                 split_distribution=None,
                 label_dict=None):
        
        """
        Args
            url :the url of the dataset
            name: name of the dataset
            backbone : the backbone to be used
            mode: the mode of the training train or test
            buffer_on: wether we bufferize the data
            save_embedding: save the embeddings on the harddisk
        
        Returns

        Raises

        
        """
        # random seeder

        # self.random_seed=31101994
        # np.random.seed(self.random_seed)
        # random.seed(self.random_seed)
        # torch.random.manual_seed(self.random_seed)

        # FILES
        # Load the files in the root folder
        self.dataset_name=name
        self.url=url
        self.files,self.Y=self.load(url,[],mode,label_dict)


        self.save_embedding=save_embedding
        self.mode=mode

        self.default_embedding_location=f"embeddings/{backbone_name}"

        
        # BACKBONE LOADING
        # Check if any backbone is given 
        # Load the right backbone in eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=torch.device(device)
        self.backbone_name=backbone_name
        


        # INCREMENTAL LEARNING
        self.split_mode=split_mode
        
        self.current_experiment=-1 # Before the beginning of the training procedure
        self.counts=np.unique(self.Y,return_counts=True)[1] # list(map(lambda x: len(os.listdir(os.path.join(url,x))),self.named_labels))
        # In the case of testing, we might have a disapearing class in case of severe inbalance,
        # in that case, we need to expand with zeros counts the non existing labels
            
        self.split_distributions=split_distribution
        if self.split_distributions is not None:
            
            if self.counts.shape[0]!=self.split_distributions.shape[0]:
                (uniqueIds,counts)=np.unique(self.Y,return_counts=True)
                n=self.split_distributions.shape[0]
                counts=np.zeros((n,1))
                for i in range(len(uniqueIds)):
                    idx=uniqueIds[i]
                    counts[idx]=self.counts[i]
                self.counts=counts

        
        
        print(f" Weights :  {self.counts}")

        
        self.weights=max(self.counts)/np.array(self.counts)
        self.n_splits=n_splits
        
        self.datasets,self.datasets_labels=self.splitDataset() # SPlit the dataset according to an industrial scheme
        self.activated_files_subset =  self.datasets[0] # The current split
        self.activated_files_labels_subset = self.datasets_labels[0] # The current split
        self.nbrs_known_classes=(self.splits[:,0]>0).sum()
        self.max_classes=len(self.splits[:,0])
        
        self.output_shape=2

        # Memory manager
        self.activated_files_subset_memory=[]
        self.activated_files_labels_subset_memory=[]

        # Current batch indexes or paths Batch indexes
        self.current_batch_paths=[]
        # Buffer to speed up the training
        self.buffer_x={}

    def step(self):
        self.current_experiment+=1
        if self.mode== "test":
        # We have a different behaviour based onn the mode we are in

            self.activated_files_subset=functools.reduce(lambda a, b: a+b, self.datasets[:self.current_experiment+1])
            self.activated_files_labels_subset=functools.reduce(lambda a, b: a+b, self.datasets_labels[:self.current_experiment+1])

        if self.mode == "train":
            self.activated_files_subset=self.datasets[self.current_experiment]
            self.activated_files_labels_subset=self.datasets_labels[self.current_experiment]

            # Add memory
            self.activated_files_subset+=self.activated_files_subset_memory
            self.activated_files_labels_subset+=self.activated_files_labels_subset_memory

    def splitDataset(self):
        """
        Split dataset according randomly to simulate an industrial continuous process
        how: cil, indus, indus_cil
        """
        # Create the splits matrix
        # n1 | n2 | ... |nn
        # .. | .. |.....|
        # f1 | f2 | ... |fn
        #  Lines sum to counts

        # def batch(iterable, n=1):
        #     l = len(iterable)
        #     for ndx in range(0, l, n):
        #         yield iterable[ndx:min(ndx + n, l)]

        
        splits=np.random.rand(self.counts.shape[0],self.n_splits)
        N_label=self.counts.shape[0]

        # Check if the split distributions exists
        if self.split_distributions is not None:
            splits=self.split_distributions


        #Otherwise, compute a new one freely
        else:

            # n_chunks=N_label//self.n_splits

            if self.split_mode == "cil":
                # CIL mode generation
                splits=np.zeros((N_label,self.n_splits),dtype=int)

                non_occupied_columns=np.arange(self.n_splits).tolist()

                for i in range(N_label):
                    if sum(splits[i,:])==0:
                        if len(non_occupied_columns)>0:
                            k=np.random.choice(non_occupied_columns)
                            splits[i,k]=1
                            non_occupied_columns.remove(k)
                        else:
                            j=np.random.choice(self.n_splits)
                            splits[i,j]=1

                        
            elif self.split_mode == "indus":
                # Pure idustrial mode
                splits=splits/np.sum(splits,axis=1).reshape(-1,1)

            elif self.split_mode == "indus_cil":
                mask= np.random.choice([True,False],size=splits.shape)
                splits=splits/np.sum(splits,axis=1).reshape(-1,1)
                splits=mask*splits+np.roll(~mask*splits,1,axis=1)
                
            
            # Save the split distribution
            # It will be passed to the eval and test set
            self.split_distributions=splits

        self.splits=splits*self.counts.reshape(-1,1)
        self.splits = self.splits.astype(np.int16)
        
        print("Splits", self.splits)

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
             mode : str,
             label_dict : Union[dict,None]):
        """
        Load on the disk the files
        It assumes that each folder represent a class.
    
        """

        #=======================================> TO MODIFY

        # Generate some dummy data and save them where the files is looking for them
        #Delete previous existing data
        
       

        # Create the folder data
        dataset_image_path = os.path.join(url,"data")
        shutil.rmtree(url)

        if not os.path.exists(dataset_image_path) : 
                os.makedirs(dataset_image_path)

                #Create the data
                n_clusters=10
                n_data=100
                x_size=2
                r_scaler=100

                alpha=np.linspace(0,1,n_clusters)
                tetha=(1-alpha)*np.pi
                r=alpha+(1-alpha)*r_scaler
                X=[]
                y=[]

                c=list(mcolors.TABLEAU_COLORS)
                datum=0.5*np.random.randn(n_data,x_size)

                for i in range(n_clusters):
                    i_dataset_image_path = os.path.join(dataset_image_path,str(i))
                    if not os.path.exists(i_dataset_image_path) : 
                            os.makedirs(i_dataset_image_path)


                for i in range(n_clusters):
                    
                    x=np.array([r[i]*np.cos(tetha[i]),r[i]*np.sin(tetha[i])])
                    x=datum+x
                    x=x/r_scaler
                    for j in range(n_data):
                        path=os.path.join(dataset_image_path,str(i),str(j))
                        np.save(path,x[j,:])


                #     plt.scatter(x[:,0],x[:,1],c=c[i])
                #     plt.grid(True)
                # plt.show()



        self.dataset_image_path = os.path.join(url,"data")
        # Get the labels names : Title of the folder

        if label_dict:
            self.label_dict=label_dict
            self.labels=list(self.label_dict.values())
            self.named_labels=list(self.label_dict.keys())

        else:
            self.named_labels=os.listdir(self.dataset_image_path)
            self.labels=np.arange(len(self.named_labels),dtype=int)
            # Create an ordered dict mapping int to 
            self.label_dict=dict(zip(self.named_labels,self.labels))
        self.label_dict_inverted=dict(zip(self.labels,self.named_labels))
        
        print(self.label_dict_inverted)

        self.rejected_classes=rejectedClasses
        
        self.n_classes=len(self.labels)
        #<======================================= END TO MODIFY

        # Load the files
        # Check if the train and test files already exists
        pre_split=os.path.exists(os.path.join(url,"train.txt"))
        if pre_split: #No : Is there a predefined files in the folder of the data . Named train.txt and test.txt ?
            train_files=np.loadtxt(os.path.join(url,"train.txt"),dtype="str")
            test_files=np.loadtxt(os.path.join(url,"test.txt"),dtype="str")
        else:

            #=============================================> TO MODIFY

            self.files,labels=self.crawlDataFolder(self.dataset_image_path,self.rejected_classes)
            #<======================================= END TO MODIFY

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
        embedding_path = path.replace('data', self.default_embedding_location)
        
        _,extension=os.path.splitext(path)
        embedding_path=embedding_path.replace(extension,".pt")
        embedding_path_folder=os.path.dirname(embedding_path)
        if not os.path.exists(embedding_path_folder): 
            os.makedirs(embedding_path_folder)
        
       

        try:
            #Try to use the embeddings
            
            x=self.buffer_x[path][0]
            y = self.buffer_x[path][1]
            #print("Using preloaded backbone")
            
        except:
           
            
            x=torch.tensor(np.load(path),dtype=torch.float32)
            
            assert x is not None
            
            #Save it in the embedding folder
            lbl=self.Y[self.files.index(path)]
            torch.save(x.detach().cpu(),embedding_path)
        
            y=torch.tensor(lbl,dtype=torch.long)

        self.current_batch_paths.append(path)
        x=torch.squeeze(x)
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
        searchedExtension =['*.jpg','*.png','*.bmp','*.jpeg','*.PNG','*.npy']
        
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
        #=======================================> TO MODIFY
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
            #Save it in the embedding folder
            
            torch.save(x.detach().cpu(),embedding_path)
        #<======================================= END TO MODIFY
        


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
        embedding_path = path.replace('data', default_embedding_location)
        
        _,extension=os.path.splitext(path)
        embedding_path=embedding_path.replace(extension,".pt")
        
        # Try to load the embeddings
        if os.path.exists(embedding_path):
            x=torch.load(embedding_path)
            
            
        else:
            img=cv2.imread(path)
            x=self.predictor.predict(img)
            #Save it in the embedding folder
            if self.predictor.backbone_name != "None":
               
                torch.save(x.detach().cpu(),embedding_path)
        
            # y=torch.tensor(lbl,dtype=torch.long)


        x=torch.squeeze(x)
        lbl=self.y[idx] 

        y=torch.tensor(int(lbl),dtype=torch.long)
        return x, y

if __name__=="__main__":

    n_splits=5


    root_folder_path="/home/mohamedphd/Documents/phd/Datasets/curated/"

    datasets_names=["dummy"]
    backbones=["None"]
    modes = ["train"]

    for backbone in backbones:
        for dataset_name in  datasets_names:
            for mode in modes:
                path=os.path.join(root_folder_path,dataset_name)
                print(f" Dataset : {dataset_name}\n Backbone : {backbone}\n mode : {mode} \n------------------------")

                dataset=DummyBaseDataset(url=path,
                                    name=dataset_name,
                                    backbone_name=backbone,
                                    mode=mode,
                                    save_embedding=True,
                                    n_splits=n_splits)
    
                for input,lbl in dataset:
                    print(input.shape)
                    break
                    
                    


