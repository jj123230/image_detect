<h1> Image detect bot </h1>
Download the whole image_detect folder as a bot, image_detect.py will be the main file to activate the bot

1. Install the env by conda env create -f folder_path\image_detect\requirement.yml
2. pip install protobuf==3.20.2 paddlepaddle paddleocr datasets==2.21.0
3. Set your host_ip, port_ip, env and DB_info in setting_config.py
4. Set your db structure in config
5. Edit the folder path in test.txt, use folder_path/image_detect/image_detect.py folder_path to activate the image_detect bot

<h2> Detect whether image is NSFW, Ads, QR Code, or inappropriate text </h2>
