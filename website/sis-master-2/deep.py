from DeepImageSearch import Load_Data, Search_Setup

image_list = Load_Data().from_folder(['static\img'])
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=1000)
st.run_index()
metadata = st.get_image_metadata_file()
abc = st.get_similar_images(image_path=image_list[0], number_of_images=10)
metadata = st.get_image_metadata_file()