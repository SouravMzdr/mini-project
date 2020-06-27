class Generate_Database:
    
    def __init__(self,model,alignment):
        import numpy as np
        import cv2
        from align import AlignDlib
        import dlib
        self.model=model
        self.database=dict()
        self.alignment=alignment
    
    
    def align_image(self,img,bb):
        alignment=self.alignment
        '''
        OBJECTIVE: Align a face based using DLIB model

        Parameter:
        test - RGB Image
        bb - bounding box around the person to be searched

        Returns:
        Aligned face

        '''

        # return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        if(bb):
            return alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        else:
            return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    
    
    
    
    def generate_encoding(self,img,bb):
        
    
        model=self.model
        '''
        OBJECTIVE: Generates an 128D encoding based on the loaded model

        Parameter:
        test - RGB Image
        bb - bounding box around the person to be searched

        Returns:
        embedded-128D encoding
        '''
        embedded = np.zeros((1, 128)) #Declare a placeholder variable
        #print(m.image_path())    
        # img = load_image(path)
        img = self.align_image(img,bb) #Align the image 
        # plt.imshow(img)
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedded= model.predict(np.expand_dims(img, axis=0))[0] #generate encoding
        return embedded
    
    
    def make_database(self,entries):
        database=self.database
       
        '''
        OBJECTIVE: Generate Database

        Parameters:python dictionary with the keys as id and values and image path 
        '''
        
        print('[INFO]:Creating New Database....')    
        for (i,j) in entries.items():
            if i not in database.keys():
                img = entries[i]
                #       print(i)
                database[i] = self.generate_encoding(img,0)
        print('[INFO]:New Database Creation Completed!')
        return database