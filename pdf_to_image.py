import os
from pdf2image import convert_from_path

def convert_pdf_to_jpg(pdf_folder = "documents", output_folder = "output_images"):
    """
    Converts each page of a pdf to image
    
    Args:
        pdf_path: 
        output_folder:
    """

    try:
        os.makedirs(output_folder, exist_ok = True)

        image_id = 0
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, filename)
                pdf_name = os.path.splitext(filename)[0]
                images = convert_from_path(pdf_path, fmt = "jpeg")
                
                for image in images:
                    image_filename = os.path.join(output_folder, f"{pdf_name}_{image_id + 1}.jpg")
                    image.save(image_filename, "JPEG")
                    print(f"Saved {image_filename}")
                    image_id += 1

        print(f"Conversion completed for {pdf_path}, saving output to {output_folder}")

    except Exception as e:
        print("Exception occured", e)
        
if __name__ == "__main__":
    convert_pdf_to_jpg()