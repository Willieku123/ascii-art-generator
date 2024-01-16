# ASCII Art Generator

![星期一清大沒有停電，交又贏](assets/nthu.jpg "星期一清大沒有停電，交又贏")


## Prepare Environment

Create a conda environment:

```
conda create -n ascii_art python=3.8
conda activate ascii_art
pip install -r requirements.txt
```

## Generate ASCII Art

Generate aN ASCII Art iamge:

```
python text2image.py --path "input/wiki.png"
```

the generated image and txt file will be saved at output/ .


Generate ASCII Art iamge in Madarin, composed of custom charactors , in dark mode, and in different resolution:
```
python text2image.py --path "input/nthu.jpg" --language "mandarin" --char "星期一清大沒有停電，交又贏" --dark --size_factor 15
```

```
                 ,       _,   /;                
               ," ',     ; - "`^                
              ;`    ""y`                        
              !.,   ,'   *      ,               
             ! `"-x.'           "\              
            !      L     !       "~,            
           /       !     !          *           
           `       "<     \          L          
          "          ,    "          !          
         "           L     ,         "          
         *           "     !         '          
 ~.~..~~.,.l._.*Z~.,(.~~,"T` :~,_,.~i~..~....~~ 
             `*_"<         '"~--~~"`            
               " "                              
               ^ !                              
              ! ;                               
              ! `                               
              ! !~:*                            
               ".-'"
```