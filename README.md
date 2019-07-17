Overview
---------
Cortex-M-KWS is an Keyword Spotting(KWS) demo runs on Cortex-M(STM32F769). It takes an quantized tensorflow model, converts it into .c and .h files, and runs the model with [Tengine-lite](https://github.com/OAID/Tengine-lite).<br>
The demo key words is "xiaozhi". The demo project can runs on STM32F769I-DISC without modify any code .<br>
The demo was developed by Keil IDE, you can build the project and download the image into the flash memory directly if you had installed Keil.<br>

Clone project
--------------
You can clone the project by ：<br>
git clone --recurse-submodules  https://github.com/OAID/cortex-m-kws.git<br>
it will clone the tengine-lite automatically<br>

Download the WAV files
---------------------------
There are 3 demo WAV files to show the Key Words， 0.wav is “xiaozhi”， 1.wav is drama part ， 2.wav is a song . Copy the 3 files into microSD , and plugin to the board <br>

Build the Project
-----------------
1.Enter the Project/MDK-ARM, then double-click the Tengine-Project icon <br>
2.build the project <br>
3.Download the code into flash memory<br>
4.The display will show the test score and result<br>
