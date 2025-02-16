import cv2
import os
import datetime

class ImageSaver:
    # -- Settings -- #
    DEFAULT_FOLDER_NAME = "Photos"

    def __init__(self, default_folder=None):
        """
        Initializes the ImageSaver.
        If a USB drive is detected, uses its path; otherwise, uses the default folder.
        """
        # Use provided folder or determine storage folder
        self.default_folder = default_folder or self.get_storage_folder()

    def get_default_folder(self):
        """
        Returns the path to the "Photos" folder in the user's Documents folder.
        """
        home_directory = os.path.expanduser("~")
        documents_directory = os.path.join(home_directory, "Documents")
        default_directory = os.path.join(documents_directory, self.DEFAULT_FOLDER_NAME)
        return default_directory

    def get_storage_folder(self):
        """
        Checks for a mounted USB flash drive and returns its path if available.
        Otherwise, returns the default folder.
        """
        # Typical mount point for USB drives on Raspberry Pi OS (e.g., /media/pi/)
        try:
            username = os.getlogin()  # e.g., "pi"
        except Exception:
            # Fallback if os.getlogin() fails
            username = os.environ.get("USER", "pi")
            
        usb_mount_root = os.path.join("/media", username)
        
        if os.path.exists(usb_mount_root):
            for entry in os.listdir(usb_mount_root):
                potential_drive = os.path.join(usb_mount_root, entry)
                if os.path.ismount(potential_drive):
                    print(f"USB drive found: {potential_drive}")
                    return potential_drive  # Use the first detected USB drive

        # If no USB drive is detected, fallback to the default folder
        print("No USB drive detected; using default folder.")
        return self.get_default_folder()

    def save_image(self, image, filename, folder=None):
        """
        Saves an image to the specified folder or the determined storage folder.
        """
        save_folder = folder or self.default_folder
        
        # Ensure the folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        now = datetime.datetime.now()
        formatted_time = now.strftime("_%Y-%m-%d_%H-%M-%S.%f")[:-3]
        file_path = os.path.join(save_folder, f"{filename}{formatted_time}.png")

        success = cv2.imwrite(file_path, image)
        print(f"Image saved: {success} to {file_path}")
        return file_path