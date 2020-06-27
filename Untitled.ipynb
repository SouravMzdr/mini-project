{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Generate_Database:\n",
    "    import cv2\n",
    "    from align import AlignDlib\n",
    "    import dlib\n",
    "    def __init__(self,model,alignment):\n",
    "        self.model=model\n",
    "        self.database=dict()\n",
    "        self.alignment=alignment\n",
    "    \n",
    "    \n",
    "    def align_image(self,img,bb):\n",
    "        alignment=self.alignment\n",
    "        '''\n",
    "        OBJECTIVE: Align a face based using DLIB model\n",
    "\n",
    "        Parameter:\n",
    "        test - RGB Image\n",
    "        bb - bounding box around the person to be searched\n",
    "\n",
    "        Returns:\n",
    "        Aligned face\n",
    "\n",
    "        '''\n",
    "\n",
    "        # return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)\n",
    "        if(bb):\n",
    "            return alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)\n",
    "        else:\n",
    "            return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)\n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    def generate_encoding(self,img,bb):\n",
    "        model=self.model\n",
    "        '''\n",
    "        OBJECTIVE: Generates an 128D encoding based on the loaded model\n",
    "\n",
    "        Parameter:\n",
    "        test - RGB Image\n",
    "        bb - bounding box around the person to be searched\n",
    "\n",
    "        Returns:\n",
    "        embedded-128D encoding\n",
    "        '''\n",
    "        embedded = np.zeros((1, 128)) #Declare a placeholder variable\n",
    "        #print(m.image_path())    \n",
    "        # img = load_image(path)\n",
    "        img = self.align_image(img,bb) #Align the image \n",
    "        # plt.imshow(img)\n",
    "        # scale RGB values to interval [0,1]\n",
    "        img = (img / 255.).astype(np.float32)\n",
    "        # obtain embedding vector for image\n",
    "        embedded= model.predict(np.expand_dims(img, axis=0))[0] #generate encoding\n",
    "        return embedded\n",
    "    \n",
    "    \n",
    "    def make_database(self,entries):\n",
    "        database=self.database\n",
    "       \n",
    "        '''\n",
    "        OBJECTIVE: Generate Database\n",
    "\n",
    "        Parameters:python dictionary with the keys as id and values and image path \n",
    "        '''\n",
    "        \n",
    "        print('[INFO]:Creating New Database....')    \n",
    "        for (i,j) in entries.items():\n",
    "            if i not in database.keys():\n",
    "                img = entries[i]\n",
    "                #       print(i)\n",
    "                database[i] = self.generate_encoding(img,0)\n",
    "        print('[INFO]:New Database Creation Completed!')\n",
    "        return database"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "db=Database(model,alignment)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "entries={myclass[i]:my_images[i] for i in range(len(myclass))}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO]:Creating New Database....\n",
      "[INFO]:New Database Creation Completed!\n"
     ]
    }
   ],
   "source": [
    "database=db.make_database(entries)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(128,)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database[\"hrishi1\"].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mini_project",
   "language": "python",
   "name": "python3.8.3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
