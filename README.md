# Facial_recognition

How to use the facial recognition python file?

1. Download the zipped repository folder.

2. Create Data path with the Computer_Camera.py file outside the Data path.

3. Take facial photos and crop the faces and create path and place them in Data/Faces/{identity} path. You can put multiple people's faces in respective identity sub paths.
   
   Example, if a person James has his photos, place them in Data/Faces/James path.

5. Ensure that yolov11n-face.pt is in Data path.

6. When the program runs, the following paths will be created:
   
   Data/Output/Frame/Known_Faces            (for recognized faces from database)
   
   Data/Output/Frame/Unknown_Faces          (for unknown faces not in database)

   Data/Output/JSON                         (json information)

   Data/Output/Log                          (logging file)

7. An example of json file data is shown below:

{

    "Number_of_faces": 1,
    
    "Faces": [
    
        {
        
            "x_min": 304,
            
            "y_min": 112,
            
            "x_max": 475,
            
            "y_max": 326,
            
            "identity": "James"
            
        }
        
    ]
    
}
