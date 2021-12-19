from urllib.request import urlretrieve
import np as np
import os
import vk_api
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA


class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image
        img = cv2.imread(self.IMAGE)

        # convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] *
                           img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_
        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def plotHistogram(self):
        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins
        =numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0),
                          (int(end), 50), (r, g, b), -1)
            start = end
        # display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()


if __name__ == "__main__":
    def auth_handler():
        """ При двухфакторной аутентификации вызывается
        эта функция.
        """
        # Код двухфакторной аутентификации
        key = input("Enter authentication code: ")
        # Если: True - сохранить, False - не сохранять.
        remember_device = True
        return key, remember_device


    login, password = '', ''
    vk_session = vk_api.VkApi(
        login, password,
        auth_handler=auth_handler  # функция
        # для
        # обработки
        # двухфакторной
        # аутентификации
    )
    try:
        vk_session.auth()
    except vk_api.AuthError as error_msg:
        print(error_msg)

    tools = vk_api.VkTools(vk_session)
    vk_app = vk_session.get_api()
    # url = input("Введите url альбома: ")
    response = tools.get_all("photos.getAll", 100, {'owner_id': -???})
    # response = tools.get_all("photos.getAll", 3)
    # работаем с каждой полученной фотографией
    for i in range(len(response["items"])):
        # берём ссылку на максимальный размер фотографии
        photo_url = str(response["items"][i]["sizes"][len(response["items"][i]["sizes"]) - 1]["url"])

        # скачиваем фото в папку с ID пользователя
        urlretrieve(photo_url, 'saved' + '/' + str(response["items"][i]['id']) + '.jpg')

    # # read image
    img = cv2.imread('C:\\Users\\artem\\PycharmProjects\\untitled7\\saved\\456239066.jpg')
    # convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get rgb values from image to 1D array
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()
    # plotting
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, g, b)
    plt.show()

    img = 'C:\\Users\\artem\\PycharmProjects\\untitled7\\saved\\456239066.jpg'
    clusters = 5
    dc = DominantColors(img, clusters)
    colors = dc.dominantColors()
    print(colors)

    dc.plotHistogram()

    data = []
    directory = 'C:\\Users\\artem\\PycharmProjects\\untitled7\\saved'
    for filename in os.listdir(directory):
        data_iter = []
        img = str(directory) + '\\' + filename
        # print (img)
        clusters = 2
        dc = DominantColors(img, clusters)
        colors = dc.dominantColors()
        for i in colors:
            for j in i:
                data_iter.append(j)
        data.append(data_iter)
        print(colors)
    np_data = np.asarray(data, dtype=np.float32)
    pca = PCA(n_components=3)
    XPCAreduced = pca.fit_transform(np_data)
    print(XPCAreduced)
    xs, ys, zs = np_data[:, 0], np_data[:, 1], np_data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
