import ctypes
import os
from tkinter import *
from tkinter import filedialog, ttk # themed tkinter module (ttk)
import cv2
import io
import numpy as np
import scipy.signal
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# np.set_printoptions(threshold=sys.maxsize)

ctypes.windll.shcore.SetProcessDpiAwareness(True)

root = Tk()
ttk.Style().configure("TButton", justify=CENTER)

# Global variables
gui_width = 1385
gui_height = 595
ip_file = ""
op_file = ""
original_img = None
modified_img = None
user_arg = None
popup = None
popup_input = None


root.title("Image processing")
# root.geometry(f"{gui_width}x{gui_height}")
# root.resizable(False, False)
root.minsize(gui_width, gui_height)


def set_user_arg():
    global user_arg
    user_arg = popup_input.get()
    popup.destroy()
    popup.quit()


def open_popup_input(text):
    global popup, popup_input
    popup = Toplevel(root) # is a widget that creates a new window on top of the current window
    popup.resizable(False, False) # prevents the window from being resized
    popup.title("User Input") # sets the title of the window
    text_label = ttk.Label(popup, text = text, justify = LEFT) # creates a label widget
    text_label.pack(side = TOP, anchor = W, padx = 15, pady = 10)
    popup_input = ttk.Entry(popup) # creates an entry field widget
    popup_input.pack(side = TOP, anchor = NW, fill = X, padx = 15)
    popup_btn = ttk.Button(popup, text = "OK", command = set_user_arg).pack(pady = 10)
    popup.geometry(f"400x{104+text_label.winfo_reqheight()}")
    popup_input.focus()
    popup.mainloop()


def draw_before_canvas():
    global original_img, ip_file
    original_img = Image.open(ip_file)
    original_img = original_img.convert("RGB")
    img = ImageTk.PhotoImage(original_img)
    before_canvas.create_image(256, 256, image = img, anchor = "center",)
    before_canvas.img = img


def draw_after_canvas(mimg):
    global modified_img
    modified_img = Image.fromarray(mimg) # creates an image object memory from the 2D array
    img = ImageTk.PhotoImage(modified_img) # this fuction allows image to be displayed in a tkinter GUI
    after_canvas.create_image(256, 256, image = img, anchor = "center") # coordinates of the image, modified image, anchor point of the image
    after_canvas.img = img


def load_file():
    global ip_file
    ip_file = filedialog.askopenfilename(
        title="Open an image file",
        initialdir=".",
        filetypes=[("All Image Files", "*.*")],)
    draw_before_canvas()
    # print(f"Image loaded from: {ip_file}")


def save_file():
    global ip_file, original_img, modified_img
    file_ext = os.path.splitext(ip_file)[1][1:]
    op_file = filedialog.asksaveasfilename(
        filetypes=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        defaultextension=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
    )
    modified_img = modified_img.convert("RGB")
    modified_img.save(op_file)
    # print(f"Image saved at: {op_file}")


# frames
left_frame = ttk.LabelFrame(root, text="Original Image", labelanchor=N)
left_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

middle_frame = ttk.LabelFrame(root, text="Algorithms", labelanchor=N)
middle_frame.pack(fill=BOTH, side=LEFT, padx=5, pady=10)

right_frame = ttk.LabelFrame(root, text="Modified Image", labelanchor=N)
right_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

# left frame contents
before_canvas = Canvas(left_frame, bg="white", width=512, height=512)
before_canvas.pack(expand=1)

browse_btn = ttk.Button(left_frame, text="Browse", command=load_file)
browse_btn.pack(expand=1, anchor=SW, pady=(5, 0))

# middle frame contents
algo_canvas = Canvas(middle_frame, width=260, highlightthickness=0)
scrollable_algo_frame = Frame(algo_canvas)
scrollbar = Scrollbar(
    middle_frame, orient="vertical", command=algo_canvas.yview, width=15
)
scrollbar.pack(side="right", fill="y")
algo_canvas.pack(fill=BOTH, expand=1)
algo_canvas.configure(yscrollcommand=scrollbar.set)
algo_canvas.create_window((0, 0), window=scrollable_algo_frame, anchor="nw")
scrollable_algo_frame.bind(
    "<Configure>", lambda _: algo_canvas.configure(scrollregion=algo_canvas.bbox("all"))
)


# right frame contents
after_canvas = Canvas(right_frame, bg="white", width=512, height=512)
after_canvas.pack(expand=1)

save_btn = ttk.Button(right_frame, text="Save", command=save_file)
save_btn.pack(expand=1, anchor=SE, pady=(5, 0))


##############################################################################

# Rizing the image
def rizing():
    image = mpimg.imread(ip_file)
    open_popup_input("Enter the new width and height of the image\nseparated by comma")
    copied_image = image.copy()
    dimension = user_arg.split(",")
    copied_image = cv2.resize(copied_image, (int(dimension[0]), int(dimension[1])))
    draw_after_canvas(copied_image)


# Convert BGR image to grayscale
def bgr_to_grayscale():
    bgr_image = mpimg.imread(ip_file)
    grayscale = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    draw_after_canvas(grayscale)


# Color Elimination (so that images can be red, blue, green, yellow, purple, or cyan..)
def color_elimination_single_channel():
    image = mpimg.imread(ip_file)
    open_popup_input("Enter the channel number to be eliminated\n0 for cyan, 1 for Purple, 2 for Yellow")
    copied_image = image.copy()
    for row in range(copied_image.shape[0]):
        for column in range(copied_image.shape[1]):
            copied_image[row, column, int(user_arg)] = 0
    draw_after_canvas(copied_image)


def color_elimination_multiple_channel():
    image = mpimg.imread(ip_file)
    open_popup_input("Enter the channel number to be displayed\n0 for red, 1 for green, 2 for blue")
    copied_image = image.copy()
    for row in range(copied_image.shape[0]):
        for column in range(copied_image.shape[1]):
            if user_arg == "0":
                copied_image[row, column, int(user_arg)+1] = 0
                copied_image[row, column, int(user_arg)+2] = 0
            elif user_arg == "1":
                copied_image[row, column, int(user_arg)-1] = 0
                copied_image[row, column, int(user_arg)+1] = 0
            else:
                copied_image[row, column, int(user_arg)-2] = 0
                copied_image[row, column, int(user_arg)-1] = 0
    draw_after_canvas(copied_image)


# Image complementing
def image_complementing():
    image = mpimg.imread(ip_file)
    modified_img = image.copy()
    modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            modified_img[row, column] = 255 - modified_img[row, column]
    draw_after_canvas(modified_img)


# Channel Swapping
def channel_swapping():
    image = mpimg.imread(ip_file)
    modified_img = image.copy()
    open_popup_input("Enter the order of channels to be swapped\nseparated by comma 0 for red, 1 for green, 2 for blue")
    channel_order = user_arg.split(",")
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            modified_img[row, column, 0] = image[row, column, int(channel_order[0])]
            modified_img[row, column, 1] = image[row, column, int(channel_order[1])]
            modified_img[row, column, 2] = image[row, column, int(channel_order[2])]
    draw_after_canvas(modified_img)


# Changing Brightness (Brighten)
def change_brightness_Brighten():
    img = cv2.imread(ip_file, 0)
    modified_img = img.copy()
    open_popup_input("Enter the value of constant")
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            if (modified_img[row, column] + int(user_arg)) > 255:
                modified_img[row, column] = 255
            else:
                modified_img[row, column] = modified_img[row, column] + int(user_arg)
    draw_after_canvas(modified_img) 


# Changing Brightness (Darken)
def change_brightness_Darken():
    img = cv2.imread(ip_file, 0)
    modified_img = img.copy()
    open_popup_input("Enter the value of constant")
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            if (modified_img[row, column] - int(user_arg)) < 0:
                modified_img[row, column] = 0
            else:
                modified_img[row, column] = modified_img[row, column] - int(user_arg)
    draw_after_canvas(modified_img) 


# Brightness color change
def brightness_color_brighter():
    img = cv2.imread(ip_file, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    modified_img = img.copy()
    open_popup_input("Enter the value of constant")
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            for channel in range(modified_img.shape[2]):
                if (modified_img[row, column, channel] + int(user_arg)) > 255:
                    modified_img[row, column, channel] = 255
                elif (modified_img[row, column, channel] + int(user_arg)) < 0:
                    modified_img[row, column, channel] = 0
                else:
                    modified_img[row, column, channel] = modified_img[row, column, channel] + int(user_arg)
    draw_after_canvas(modified_img)    


def brightness_color_darker():
    img = cv2.imread(ip_file, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    modified_img = img.copy()
    open_popup_input("Enter the value of constant")
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            for channel in range(modified_img.shape[2]):
                if (modified_img[row, column, channel] - int(user_arg)) > 255:
                    modified_img[row, column, channel] = 255
                elif (modified_img[row, column, channel] - int(user_arg)) < 0:
                    modified_img[row, column, channel] = 0
                else:
                    modified_img[row, column, channel] = modified_img[row, column, channel] - int(user_arg)
    draw_after_canvas(modified_img) 


# Brightness Color Change (Red, Green, Blue, Yellow, Purple, Cyan)
def brightness_color():
    img = cv2.imread(ip_file, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    modified_img = img.copy()
    open_popup_input("Enter the value of constant and the channel number\n0 for red, 1 for green, 2 for blue separated by comma")
    user_arg_list = user_arg.split(",")
    for row in range(modified_img.shape[0]):
        for column in range(modified_img.shape[1]):
            if len(user_arg_list) == 2:
                if (modified_img[row, column, int(user_arg_list[1])] + int(user_arg_list[0])) > 255:
                    modified_img[row, column, int(user_arg_list[1])] = 255
                else:
                    modified_img[row, column, int(user_arg_list[1])] = modified_img[row, column, int(user_arg_list[1])] + int(user_arg_list[0])
            else:
                if (modified_img[row, column, int(user_arg_list[1])] + int(user_arg_list[0])) > 255:
                    modified_img[row, column, int(user_arg_list[1])] = 255
                else:
                    modified_img[row, column, int(user_arg_list[1])] = modified_img[row, column, int(user_arg_list[1])] + int(user_arg_list[0])
                
                if (modified_img[row, column, int(user_arg_list[2])] + int(user_arg_list[0])) > 255:
                    modified_img[row, column, int(user_arg_list[2])] = 255
                else:
                    modified_img[row, column, int(user_arg_list[2])] = modified_img[row, column, int(user_arg_list[2])] + int(user_arg_list[0])
    
    draw_after_canvas(modified_img)


# Histogram Stretching
def histogram_stretching():
    image = cv2.imread(ip_file)
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the minimum and maximum pixel values
    min_val = np.min(grayscale)
    max_val = np.max(grayscale)
    # Perform histogram stretching
    stretched = (grayscale - min_val) / (max_val - min_val) * 255
    # Convert floating point image to uint8
    stretched = np.uint8(stretched)
    draw_after_canvas(stretched)


# Histogram Equalization
def histogram_equalization():
    image = cv2.imread(ip_file)
    img_eq = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    draw_after_canvas(img_eq)


# Calculate Histogram
def plot_colored_histogram():
    image = cv2.imread(ip_file)
    histogram = [[0]*256, [0]*256, [0]*256] 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for channel in range(image.shape[2]):
                i = image[i,j, channel]
                histogram[channel][i] = histogram[channel][i]+1
    colors = ["blue", "green", "red"] 
    for i in range(image.shape[2]): 
        plt.plot(histogram[i], color = colors[i])
    # plt.show()
    plt.title(f'Histogram for colored image')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = Image.open(buf)
    plt.close()
    draw_after_canvas(np.array(plot_image))


# Calculate Histogram of a Channel
def plot_channel_histogram():
    image = cv2.imread(ip_file)
    open_popup_input("Enter the channel number 0 for red, 1 for green, 2 for blue")
    histogram = [0]*256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = image[i,j, int(user_arg)]
            histogram[val] = histogram[val]+1
    colors = ["red", "green", "blue"] 
    plt.plot(histogram, color = colors[int(user_arg)])
    plt.title(f'Histogram for {colors[int(user_arg)]} channel')
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = Image.open(buf)
    plt.close()
    draw_after_canvas(np.array(plot_image))


# Calculate Grayscale Histogram
def plot_grayscale_histogram():
    image = cv2.imread(ip_file)
    # Convert the image to grayscale if it is not already
    if len(image.shape) > 2:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image
    # Calculate the histogram
    histogram = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    # Plot the histogram
    plt.plot(histogram, color='k')
    plt.title('Histogram for grayscale image')
    plt.xlim([0, 256])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = Image.open(buf)
    plt.close()
    draw_after_canvas(np.array(plot_image))
    # plt.show()


def average_filter_5x5():
    # Loading the image
    image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    # Create a 5x5 averaging mask
    mask = np.ones([5, 5], dtype=np.float32) / 25
    # rows and columns of the image
    rows, columns = image.shape[0], image.shape[1]
    # creating 2D array with same dimesions as image
    new_image = np.zeros([rows, columns])
    # calculating the average filter for each pixel
    for i in range(2, rows - 2):
        for j in range(2, columns - 2):
            average = image[i-2, j-2] * mask[0, 0] + image[i-2, j-1] * mask[0, 1] + image[i-2, j] * mask[0, 2] + image[i-2, j+1] * mask[0, 3] + image[i-2, j+2] * mask[0, 4] + image[i-1, j-2] * mask[1, 0] + image[i-1, j-1] * mask[1, 1] + image[i-1, j] * mask[1, 2] + image[i-1, j+1] * mask[1, 3] + image[i-1, j+2] * mask[1, 4] + image[i, j-2] * mask[2, 0] + image[i, j-1] * mask[2, 1] + image[i, j] * mask[2, 2] + image[i, j+1] * mask[2, 3] + image[i, j+2] * mask[2, 4] + image[i+1, j-2] * mask[3, 0] + image[i+1, j-1] * mask[3, 1] + image[i+1, j] * mask[3, 2] + image[i+1, j+1] * mask[3, 3] + image[i+1, j+2] * mask[3, 4] + image[i+2, j-2] * mask[4, 0] + image[i+2, j-1] * mask[4, 1] + image[i+2, j] * mask[4, 2] + image[i+2, j+1] * mask[4, 3] + image[i+2, j+2] * mask[4, 4]
            new_image[i, j] = 255 if average > 255 else average
    new_image = new_image.astype(np.uint8)
    # Display the original and average filter images
    draw_after_canvas(new_image)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Average Filter Image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def range_filter():
    # Loading the image
    image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    rows, columns = image.shape[0], image.shape[1]
    new_image = np.zeros([rows, columns])
    for i in range(1, rows-1): 
        for j in range(1, columns-1): 
            neighborhood = [image[i-1, j-1], image[i-1, j], image[i-1, j+1], 
                             image[i, j-1], image[i, j], image[i, j+1], 
                             image[i+1, j-1], image[i+1, j], image[i+1, j+1]]
            new_image[i, j] = np.max(neighborhood) - np.min(neighborhood)
    new_image = new_image.astype(np.uint8)
    draw_after_canvas(new_image)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Range Filter Image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def min_filter_3x3():
    # Loading the image
    image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    rows, columns = image.shape[0], image.shape[1]
    new_image = np.zeros([rows, columns])
    for i in range(1, rows-1):
        for j in range(1, columns-1): 
            neighborhood = [image[i-1, j-1], image[i-1, j], image[i-1, j+1], 
                             image[i, j-1], image[i, j], image[i, j+1], 
                             image[i+1, j-1], image[i+1, j], image[i+1, j+1]]
            new_image[i, j] = np.min(neighborhood)
    new_image = new_image.astype(np.uint8)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Range Filter Image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    draw_after_canvas(new_image)


def max_filter_3x3():
    # Loading the image
    image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    rows, columns = image.shape[0], image.shape[1]
    new_image = np.zeros([rows, columns])
    for i in range(1, rows-1):
        for j in range(1, columns-1): 
            neighborhood = [image[i-1, j-1], image[i-1, j], image[i-1, j+1], 
                             image[i, j-1], image[i, j], image[i, j+1], 
                             image[i+1, j-1], image[i+1, j], image[i+1, j+1]]
            new_image[i, j] = np.max(neighborhood)
    new_image = new_image.astype(np.uint8)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Range Filter Image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    draw_after_canvas(new_image)


def medien_filter():
    # Loading the image
    image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    rows, columns = image.shape[0], image.shape[1]
    new_image = np.zeros([rows, columns])
    for i in range(1, rows-1): 
        for j in range(1, columns-1): 
            neighborhood = [image[i-1, j-1], image[i-1, j], image[i-1, j + 1], 
            image[i, j-1], image[i, j], image[i, j + 1], 
            image[i + 1, j-1], image[i + 1, j], image[i + 1, j + 1]]
            new_image[i, j] = np.median(neighborhood)
    new_image = new_image.astype(np.uint8)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Range Filter Image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    draw_after_canvas(new_image)

# Morphological Operations

def dilate():
    original_image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    image_height, image_width = binary_image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(binary_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    result = np.zeros_like(binary_image)

    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            roi = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            result[i - pad_height, j - pad_width] = np.max(roi * kernel)

    draw_after_canvas(result) 


def erode():
    original_image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    image_height, image_width = binary_image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(binary_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)

    result = np.zeros_like(binary_image)

    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            roi = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            result[i - pad_height, j - pad_width] = np.min(roi * kernel)

    draw_after_canvas(result)


def opening():
    eroded_image = erode()
    kernel = np.ones((3, 3), dtype=np.uint8)
    image_height, image_width = eroded_image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(eroded_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    result = np.zeros_like(eroded_image)

    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            roi = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            result[i - pad_height, j - pad_width] = np.max(roi * kernel)

    draw_after_canvas(result)

def closing():
    dilated_image = dilate()
    kernel = np.ones((3, 3), dtype=np.uint8)
    image_height, image_width = dilated_image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(dilated_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)

    result = np.zeros_like(dilated_image)

    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            roi = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            result[i - pad_height, j - pad_width] = np.min(roi * kernel)

    draw_after_canvas(result)


def global_threshold():
    original_image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    threshold_value = 128
    _, binary_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)
    draw_after_canvas(binary_image)

def adaptive_threshold():
    original_image = cv2.imread(ip_file, cv2.IMREAD_GRAYSCALE)
    block_size = 11
    binary_image = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)
    draw_after_canvas(binary_image)


"""
->  it uses the ttk (themed tkinter) module, which provides access to the Tk themed widget set.

->  The `pack` method is then called on the button object to place it in the GUI.
    The `expand`, `padx`, `pady`, and `ipady` parameters control how the button is placed
    and how it interacts with resizing of the GUI. `expand=1` allows the widget to expand
    to fill any extra space in the GUI. `padx` and `pady` add padding (extra space) around the widget,
    and `ipady` adds internal padding (extra space) inside the widget.

"""
# Fucntionlty buttons

ttk.Button(
            scrollable_algo_frame, 
            text = "Resize the image", # text to be displayed on the button
            width = 30, # sets the width of the button to 30 characters
            command = rizing # calls the function callRGB2Gray when clicked
            ).pack(expand = 1, padx = 5, pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text = "BGR to Grayscale", # text to be displayed on the button
            width = 30, # sets the width of the button to 30 characters
            command = bgr_to_grayscale # calls the function callRGB2Gray when clicked
            ).pack(expand = 1, padx = 5, pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Color Elimination \n(Single Channel)",
            width = 30,
            command = color_elimination_single_channel
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Color Elimination\n(Multiple Channels)",
            width = 30,
            command = color_elimination_multiple_channel
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame,
            text = "Complementing",
            width = 30,
            command = image_complementing
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame,
            text = "Swapping Channels",
            width = 30,
            command = channel_swapping
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= " Changing Brightness Grayscale\n(Brighter)", 
            width = 30, 
            command = change_brightness_Brighten
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= " Changing Brightness Grayscale\n(Darker)", 
            width = 30, 
            command = change_brightness_Darken
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= " Changing Brightness Colored\n(Brighter)", 
            width = 30, 
            command = brightness_color_brighter
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= " Changing Brightness Colored\n(Darker)", 
            width = 30, 
            command = brightness_color_darker
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= " Changing Brightness\nSpecific Channel", 
            width = 30, 
            command = brightness_color
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Histogram Stretching", 
            width = 30, 
            command = histogram_stretching
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Histogram Equalization", 
            width = 30, 
            command = histogram_equalization
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Calculate Colored\nHistogram", 
            width = 30, 
            command = plot_colored_histogram
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Calculate Histogram\n of a Channel", 
            width = 30, 
            command = plot_channel_histogram
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Calculate Grayscale\nHistogram", 
            width = 30,
            command = plot_grayscale_histogram
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Average Filter\n(5x5)", 
            width = 30,
            command = average_filter_5x5
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Range Filter", 
            width = 30,
            command = range_filter
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Min Filter\n(3x3)",
            width = 30,
            command = min_filter_3x3
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Max Filter\n(3x3)",
            width = 30,
            command = max_filter_3x3
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Medien Filter",
            width = 30,
            command = medien_filter
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Dilate",
            width = 30,
            command = dilate
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Erode",
            width = 30,
            command = erode
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Opening",
            width = 30,
            command = opening
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Closing",
            width = 30,
            command = closing
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Global Thresholding",
            width = 30,
            command = global_threshold
            ).pack(pady = 2, ipady = 2)


ttk.Button(
            scrollable_algo_frame, 
            text= "Adaptive Thresholding",
            width = 30,
            command = adaptive_threshold
            ).pack(pady = 2, ipady = 2)


root.mainloop()