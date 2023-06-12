
import cv2
import numpy as np
import os.path as osp
import os
from copy import deepcopy
import random
import warnings
warnings.filterwarnings("ignore")#, "Corrupt EXIF data", UserWarning)
import pymfe
from pymfe.mfe import MFE

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
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class nouvelOpDataset(torch.utils.data.Dataset):
    def __init__(self, datasetPath,
                 datasetName="nouvelop",
                 rejectedClasses=[],
                 extension='png',
                 mode="train",
                 level=2,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/",
                 n_experiments=2,
                 device="cpu"):
        self.dataset_name=datasetName
        self.saving_folder=saving_folder+self.dataset_name
        self.level=level
        self.n_splits=n_experiments
        self.current_experiment=-1 # Before the beginning of the training procedure



        # Load the sam VIT Network , extract the features dans save them for further use

        sam_checkpoint = "./sam_vit/sam_vit_b_01ec64.pth"
        model_type = "vit_b"


        model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor=SamPredictor(model)
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=torch.device(device)
        
        self.model=model
        self.model.to(device=device)
        
        self.load(datasetPath,rejectedClasses,extension,model,level)
        self.Y=list(map(lambda x:self.label_dict[x.split('/')[-1*level]],self.files)) # Get ALL the numeric label
        self.Y=np.array(self.Y)
        self.counts=np.unique(self.Y,return_counts=True)[1] # list(map(lambda x: len(os.listdir(os.path.join(datasetPath,x))),self.named_labels))
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


        # Curent batch indexes or paths Batch indexes
        self.current_batch_paths=[]

        # Buffer to speed up the training
        self.buffer_x={}
        





    def batch_sam_preprocess(self,images):
        """
        This function preprocess a batch of images so that the vit-H tranformer of SAm model can generate embeddings
        Image format is rgb and are numpy arrays

        """

        # Transform the image to the form expected by the model
        target_length=self.model.image_encoder.img_size
        image_shape=images.shape # bs, H,W, C
        oldh=image_shape[1]
        oldw=image_shape[2]
        long_side_length=target_length

        

        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        target_size=(newh,neww)
        new_image_shape=(image_shape[0],image_shape[-1],newh,neww)

        x_=torch.tensor(images,dtype=torch.float32) # Size is in form (bs,h,w, c)
        x_=x_.permute(0,3,1,2)
        input_images=T.Resize(target_size)(x_)
        
        return input_images
        
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



    def load(self,datasetPath,rejectedClasses,extension,mode,level):
        self.dataset_image_path = datasetPath
        self.named_labels=os.listdir(self.dataset_image_path)
        self.labels=np.arange(len(self.named_labels),dtype=float)
        self.label_dict=dict(zip(self.named_labels,self.labels))
        # with open(os.path.join(self.saving_folder,"labels_dict.json"),"w") as f:
        #     json.dump(self.label_dict,f)
        self.rejected_classes=rejectedClasses
        self.n_classes=len(self.labels)
        self.files,_=self.crawlDataFolder(self.dataset_image_path,extension,self.rejected_classes)

        # Use the mode to generate the splits ( )
        alpha=0.8
        split_idx=int(alpha*len(self.files))
        if mode=="train":
            self.files=self.files[:split_idx]
        elif mode=="val":
            self.files=self.files[split_idx:]



        # self.X,self.Y=self.generateEmbeddings(model,self.files,level=level, savingFolder=self.saving_folder,bs=2)
            
    
    def generateMetafeatures(self,force_regeneration=False):
        saveName=os.path.join(self.saving_folder,"dataset_metafeatures_dict.json")
        if (os.path.exists(saveName)) and (not force_regeneration):
            with open(saveName,"r") as f:
                self.datasetFeatures_dict=json.load(f)
                ft=[list(self.datasetFeatures_dict.keys()),list(self.datasetFeatures_dict.values())]
                self.datasetFeatures=ft
                print("\n".join("{:50}  {:30}".format(x,y) for x,y in zip(ft[0],ft[1])))
                return

        self.mfe=MFE(groups="all",
                     summary=["mean"],
                     features=["ch","sil","vdb","vdu","c1","c2","l3","l2","linear_discr","naive_bayes","elite_nn","p_trace"])
            
        self.mfe.fit(self.X,self.Y)
        ft=self.mfe.extract()
        print("\n".join("{:50}  {:30}".format(x,y) for x,y in zip(ft[0],ft[1])))
        self.datasetFeatures=ft
        self.datasetFeatures_dict=dict(zip(ft[0],ft[1]))
        with open(os.path.join(self.saving_folder,"dataset_metafeatures_dict.json"),"w") as f:
            json.dump(self.datasetFeatures_dict,f)


    def __len__(self):
        # print(f"Length of the dataloader is {len(self.activated_files_subset)}")
        
        return len(self.activated_files_subset)

    def __getitem__(self, idx):
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

    def crawlDataFolder(self,entryDirectory,searchedExtension="npy",reject=[]):
        """
        Look in all subfolders for any files with the requested extension
        """
        foundFilesPaths=[]
        foundFilesNames=[]
        foundFilesLabels=[]
        for root, dir_names, file_names in os.walk(entryDirectory):
            
            for f in file_names:
                rejections=list(map(lambda x: x in root,reject))
                if any(rejections):
                    continue

                fname = os.path.join(root, f)
                if fname.endswith(searchedExtension):
                    foundFilesPaths.append(fname)
                    foundFilesNames.append(f)
                    foundFilesLabels.append(fname.replace(searchedExtension,"txt"))
        return foundFilesPaths,foundFilesNames
    
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

        y=torch.tensor(lbl,dtype=torch.long)
        return x, y

    



class tiles(nouvelOpDataset):
    '''
    Magnetic-tile-defect-datasets.-master


    The folder nems are the names of the labels
    THe jpg files are images
    the png are the labels
    
    '''
    def __init__(self, datasetPath,
                 datasetName="MagneticTile",
                 rejectedClasses=[],
                 extension='jpg',
                 mode="train",
                 level=3,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/",
                 n_experiments=2,device="cpu"):
        super().__init__(datasetPath,
                 datasetName,
                 rejectedClasses,
                 extension,
                 mode,
                 level,
                 saving_folder,n_experiments,device)



class kaggleSynthetic(nouvelOpDataset):
    '''
    dagm_kaggleupload=
    '''
    def __init__(self, datasetPath,
                 datasetName="kaggleTextureSynthetic",
                 rejectedClasses=[],
                 extension='PNG',
                 mode="train",
                 level=3,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/"):
        super().__init__(datasetPath,
                 datasetName,
                 rejectedClasses,
                 extension,
                 mode,
                 level,
                 saving_folder)


class mvtec(nouvelOpDataset):
    '''
    mvtec
    '''
    def __init__(self, datasetPath,
                 datasetName="mvtec",
                 rejectedClasses=[],
                 extension='png',
                 mode="train",
                 level=4,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/",
                 n_experiments=2,device="cpu"):
        super().__init__(datasetPath,
                 datasetName,
                 rejectedClasses,
                 extension,
                 mode,
                 level,
                 saving_folder,
                 n_experiments,device)
        

class severstal(nouvelOpDataset):
    '''
    severstal
    This dataset needs a particular loading using a csv load
    WE load only the train folder since we do have a train folder
    '''
    def __init__(self, datasetPath,
                 datasetName="severstal",
                 rejectedClasses=[],
                 extension='jpg',
                 mode="train",
                 level=4,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/"):
        super().__init__(datasetPath,
                 datasetName,
                 rejectedClasses,
                 extension,
                 mode,
                 level,
                 saving_folder)
    

    def load(self,datasetPath,rejectedClasses,extension,model,level):
        self.dataset_image_path = datasetPath
        import pandas as pd
        df=pd.read_csv(os.path.join(self.dataset_image_path,"train.csv"))
        #self.named_labels=os.listdir()
        self.labels=pd.unique(df["ClassId"])
        self.labels=self.labels.astype(float)
        self.named_labels=list(map(lambda x: str(x),self.labels.tolist()))
            #np.arange(len(self.named_labels))
        self.label_dict=dict(zip(self.named_labels,self.labels))
        self.rejected_classes=rejectedClasses
        self.n_classes=len(self.labels)
        self.df=dict(zip(df["ImageId"],df["ClassId"]))
        with open(os.path.join(self.saving_folder,"labels_dict.json"),"w") as f:
            json.dump(self.label_dict,f)
        self.files,_=self.crawlDataFolder(os.path.join(self.dataset_image_path,"train_images"),extension,self.rejected_classes)
                    #Ensure all images has a labels
        files=list(map(lambda x:x.split('/')[-1],self.files))
        files=set((self.df.keys())) & set(files)
        self.files=list(map(lambda x:os.path.join(os.path.join(self.dataset_image_path,"train_images"),x),files))
        self.X,self.Y=self.generateEmbeddings(model,self.files, savingFolder=self.saving_folder,level=level,bs=1024)
            
    

    def generateEmbeddings(self,model,files, savingFolder,level=3,bs=1024):
        """
        This function generates embeddings for the whole dataset at once and saves it in a folder
        """
        
        X= np.zeros((len(files),512))
        Y= np.zeros((len(files)),dtype=np.uint8)



        for i in range(0,len(files),bs):
            imgs=list(map(lambda x: cv2.resize(cv2.imread(x), (224,244), interpolation = cv2.INTER_NEAREST),files[i:i+bs]))
            try:
                lbs=list(map(lambda x:self.df[x.split('/')[-1]],files[i:i+bs])) # Get the numeric label 
            except:
                lbs=list(map(lambda x:x.split('/')[-1*level],files[i:i+bs])) # Get the numeric label 
                print(lbs)
                raise KeyError
            x_=np.array(imgs)
            x_=x_.transpose(0,3,1,2)/255.0
            x_=(x_- self.mean)/self.std
            x_=torch.tensor(x_,dtype=torch.float32).to(self.device)
            with torch.no_grad():
                x_embeddings=model(x_) # bsx512
                embeddings=x_embeddings.cpu().numpy()
                X[i:i+bs,:]=embeddings
                Y[i:i+bs]=np.array(lbs)

        
        # Save in a folder
        if not os.path.exists(savingFolder):
            # Create the folders
            os.makedirs(savingFolder)
            os.makedirs(os.path.join(savingFolder,"data"))
            os.makedirs(os.path.join(savingFolder,"labels"))

        np.save(os.path.join(savingFolder,"data/X.npy"),X)
        np.save(os.path.join(savingFolder,"labels/y.npy"),Y)
        return X,Y
            # random.seed(10)



class eaydataset(nouvelOpDataset):
    '''
    hazelnut beacon coffee
    '''
    def __init__(self, datasetPath,
                 datasetName="easydataset",
                 rejectedClasses=[],
                 extension='jpg',
                 mode="train",
                 level=4,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/",
                 n_experiments=2,device="cpu"):
        super().__init__(datasetPath,
                 datasetName,
                 rejectedClasses,
                 extension,
                 mode,
                 level,
                 saving_folder,n_experiments,device)

class kth(tiles):
    '''
    TEXTURES
    '''
    def __init__(self, datasetPath,
                 datasetName="kth",
                 rejectedClasses=[],
                 extension='png',
                 mode="train",
                 level=3,
                 saving_folder="/home/mohamedphd/Documents/phd/embeddings/",
                 n_experiments=2,device="cpu"):

        super().__init__(datasetPath,
                        datasetName,
                        rejectedClasses,
                        extension,
                        mode,
                        level,
                        saving_folder,n_experiments,device)

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

