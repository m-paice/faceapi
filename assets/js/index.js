const cam = document.getElementById("cam");

const startVideo = () =>
  navigator.getUserMedia(
    { video: true },
    stream => (cam.srcObject = stream),
    error => console.error(error)
  );

const loadLabels = () => {
  const labels = ["Matheus Paice"];
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 5; i++) {
        const img = await faceapi.fetchImage(
          `/assets/lib/labels/${label}/${i}.jpeg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
};

// redes reurais
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/assets/lib/models"), // detectar rostos
  faceapi.nets.faceLandmark68Net.loadFromUri("/assets/lib/models"), // desenhar os traços do rosto
  faceapi.nets.faceRecognitionNet.loadFromUri("/assets/lib/models"), // reconhecimento da pessoa
  faceapi.nets.faceExpressionNet.loadFromUri("/assets/lib/models"), // detectar expressões
  faceapi.nets.ageGenderNet.loadFromUri("/assets/lib/models"), // detectar idade e genero
  faceapi.nets.ssdMobilenetv1.loadFromUri("/assets/lib/models") // desenhar o quadrado na tela
]).then(startVideo);

cam.addEventListener("play", async () => {
  const canvas = faceapi.createCanvasFromMedia(cam);
  const canvasSize = {
    width: cam.width,
    height: cam.height
  };

  const labels = await loadLabels();

  faceapi.matchDimensions(canvas, canvasSize);
  document.body.appendChild(canvas);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(cam, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, canvasSize);
    const faceMatcher = new faceapi.FaceMatcher(labels, 0.6);
    const results = resizedDetections.map(d =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
    resizedDetections.forEach(detection => {
      const { age, gender, genderProbability } = detection;
      new faceapi.draw.DrawTextField(
        [
          `${Math.floor(age)} anos`,
          `${gender} (${Math.floor(genderProbability * 100)} %)`
        ],
        detection.detection.box.topRight
      ).draw(canvas);
    });
    results.forEach((result, index) => {
      const box = resizedDetections[index].detection.box;
      const { label, distance } = result;
      new faceapi.draw.DrawTextField(
        [`${label} (${Math.floor(distance * 100)} %)`],
        box.bottomRight
      ).draw(canvas);
    });
  }, 100);
});
