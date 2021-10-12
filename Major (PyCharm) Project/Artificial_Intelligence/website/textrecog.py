def rectify(h):
        h = h.reshape((4,2))
        hnew = numpy.zeros((4,2),dtype = numpy.float32)

        add = h.sum(1)
        hnew[0] = h[numpy.argmin(add)]
        hnew[2] = h[numpy.argmax(add)]

        diff = numpy.diff(h,axis = 1)
        hnew[1] = h[numpy.argmin(diff)]
        hnew[3] = h[numpy.argmax(diff)]

        return hnew

    def resize_image(image,width,height):
        image = cv2.resize(image,(width,height))
        return image

    def gray_image(image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image

    def canny_edge_detection(image):
        blurred_image = cv2.GaussianBlur(image, (5,5), 0)
        edges = cv2.Canny(blurred_image,0,50)
        return edges

    def find_contours(image):
        (contours, _) = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                target = approx
                break
        return target

    def draw_contours(orig_image, image, target):
        approx = rectify(target)
        pts2 = numpy.float32([[0,0],[800,0],[800,800],[0,800]])

        M = cv2.getPerspectiveTransform(approx,pts2)
        result = cv2.warpPerspective(image,M,(800,800))

        cv2.drawContours(image, [target], -1, (0,255,0),2)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        return result

    image = cv2.imread(myimg)
    image = resize_image(image,1500,800)
    orig = image
    # cv2.imshow(orig)

    gray = gray_image(image)
    # cv2_imshow(gray)

    edges = canny_edge_detection(gray)
    # cv2.imshow(edges)

    target = find_contours(edges)
    output = draw_contours(orig,image,target)
    text = pytesseract.image_to_string(output)

    return render(request, "processtext.html", {'text': text})