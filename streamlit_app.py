import streamlit as st
from gradio_client import Client, file
from PIL import Image
import tempfile
import time
import os
import base64

# Initialize the Gradio clients
client_sdxl = Client("MotionDiz/SDXL-Turbo-Img2Img-CPU")
client_instant_mesh = Client("TencentARC/InstantMesh")

def convert_image(image):
    seed = client_sdxl.predict(api_name="/get_random_value")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file, format='PNG')
        temp_file_path = temp_file.name
    
    # Use the default prompt
    prompt = "A front-facing image of an animal posing for the camera, 3D talking tom style, full body standing"
    
    result = client_sdxl.predict(
        source_img=file(temp_file_path),
        prompt=prompt,
        steps=5,  # Default steps
        seed=seed,
        Strength=0.5,  # Default strength
        api_name="/predict"
    )
    
    return Image.open(result)

def generate_3d_mesh(image_path):
    # Check input image
    client_instant_mesh.predict(
        input_image=file(image_path),
        api_name="/check_input_image"
    )

    # Preprocess
    client_instant_mesh.predict(
        input_image=file(image_path),
        do_remove_background=True,
        api_name="/preprocess"
    )

    # Generate MVS
    client_instant_mesh.predict(
        input_image=file(image_path),
        sample_steps=75,
        sample_seed=42,
        api_name="/generate_mvs"
    )

    # Make 3D
    result = client_instant_mesh.predict(
        api_name="/make3d"
    )
    
    # Check if result is a tuple and extract the file path
    if isinstance(result, tuple):
        obj_file = result[0]  # Assume the first element is the file path
    else:
        obj_file = result

    if not isinstance(obj_file, (str, bytes, os.PathLike)):
        raise ValueError(f"Expected a file path, but got {type(obj_file)}")

    return obj_file

def display_obj(obj_path):
    with open(obj_path, "r") as file:
        obj_text = file.read()
    
    obj_base64 = base64.b64encode(obj_text.encode()).decode()
    
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
    <div id="3d-container" style="width:100%;height:400px;"></div>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, 400);
        document.getElementById('3d-container').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        
        const loader = new THREE.OBJLoader();
        const objData = atob('{obj_base64}');
        const object = loader.parse(objData);
        scene.add(object);
        
        camera.position.z = 5;
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    
    st.components.v1.html(html, height=400)

# Streamlit app
st.title("Image-to-3D Conversion")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Convert image
        converted_image = convert_image(image)
        st.image(converted_image, caption="Converted Image", use_column_width=True)
        
        # Save converted image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            converted_image.save(temp_file, format='PNG')
            temp_file_path = temp_file.name
        
        # Generate 3D mesh
        obj_file = generate_3d_mesh(temp_file_path)
        
        status_text.text("3D mesh generated successfully!")
        
        # Check if obj_file exists and is readable
        if os.path.isfile(obj_file) and os.access(obj_file, os.R_OK):
            # Display the OBJ file
            st.subheader("3D Model Viewer")
            display_obj(obj_file)
            
            # Provide download link for the OBJ file
            with open(obj_file, "rb") as file:
                btn = st.download_button(
                    label="Download OBJ file",
                    data=file,
                    file_name="model.obj",
                    mime="model/obj"
                )
        else:
            st.error(f"Unable to read the generated OBJ file: {obj_file}")
    
    except Exception as e:
        status_text.text(f"An error occurred: {str(e)}")
    finally:
        progress_bar.empty()

st.markdown("Note: This app uses the SDXL-Turbo-Img2Img-CPU and InstantMesh models from Hugging Face Spaces.")