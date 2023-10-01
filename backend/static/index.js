window.addEventListener("DOMContentLoaded", () => {
	document.getElementById('upload-images').addEventListener("change", uploadImages);
	document.getElementById('select-images').addEventListener("click", selectImages);

	document.getElementById('download-csv').addEventListener("click", () => {
		fetch('/predicts/csv')
	});
	document.getElementById('download-archive').addEventListener("click", () => {
		fetch('/predicts/archive')
	});
	uploadTimeValue = document.getElementById('upload-time-value')
	savedTimeValue = document.getElementById('saved-time-value')

	$(function() {
		//hang on event of form with id=myform
		$("#upload-images-form").submit(function(e) {
			e.preventDefault();    
			var formData = new FormData(this);
			console.log(formData)

			$.ajax({
				url: "/upload-images",
				type: 'POST',
				data: formData,
				success: function (data) {
					console.log(data)
				},
				error: function (data) {
					console.err(data)
				},
				cache: false,
				contentType: false,
				processData: false
			});
			
		});
	});
});


function selectImages() {
	document.getElementById("upload-images").click();
}

function uploadImages({target}) {
	console.log('Uploading images')
	document.getElementById('upload-images-form').style.height = '50px';
	document.getElementById('result-window').style.display = 'block';
	document.getElementById('saved-time').style.display = 'block';
	document.getElementById('download-buttons').style.bottom = '0';
	uploadFile();
}

function uploadFile(){
  let xhr = new XMLHttpRequest();
	const session = "asdfasdfdaf";
	const startTime = new Date().getTime() / 1000;

  xhr.open("POST", "/upload-images");
	xhr.setRequestHeader("session", session);
  xhr.upload.addEventListener("progress", ({loaded, total}) =>{
    let fileLoaded = Math.floor((loaded / total) * 100);
		uploadTimeValue.textContent = `Загружено ${fileLoaded}%`;
		const elapsedTime = new Date().getTime() / 1000 - startTime;
		const elapsedInMinutes = Math.round(elapsedTime * 3 / 60);
		let measurement = 'минут';
		if (elapsedInMinutes == 1) {
			measurement = 'минуту'
		} else if (elapsedInMinutes >= 2 && elapsedInMinutes <= 4) {
			measurement = 'минуты'
		}
		savedTimeValue.textContent = `${elapsedInMinutes} ${measurement}`;

  });
	xhr.responseType = 'text'
	xhr.onreadystatechange = function() {
		if (xhr.readyState == 4 && xhr.status == 200) {
			console.log('Final result: ' + xhr);
			document.getElementById('download-buttons').style.opacity = '1';
			document.getElementById('job-done').style.opacity = '1';
		}
	}
	let data = new FormData(document.getElementById('upload-images-form'));
	let totalFiles = 0
	data.forEach(() => totalFiles += 1)
	console.log('Total files: ' + totalFiles)
  xhr.send(data);

	const brokenBar = document.getElementById('broken-bar')
	const emptyBar = document.getElementById('empty-bar')
	const animals = [
		document.getElementById('bird-bar'),
		document.getElementById('mammal-bar'),
	];

	const stepSize = 1 / totalFiles;
	console.log('Step size: ' + stepSize)

	fetch('/predicts', {headers: {Session: session}})
		.then(response => response.body)
		.then(rs => {
			const reader = rs.getReader();
			return new ReadableStream({
				async start() {
					while (true) {
						const { done, value } = await reader.read();
						if (done) {
							break;
						}
						const jsonString = new TextDecoder().decode(value)
						const parsedData = JSON.parse(jsonString)
						console.log(parsedData)
						parsedData.predict.forEach(p => {
							fileName = p[0]
							is_broken = p[1]
							is_empty = p[2]
							let toIncrement;
							if (is_broken == 1) {
								toIncrement = brokenBar;
							} else if (is_empty == 1) {
								toIncrement = emptyBar;
							} else {
								toIncrement = animals[Math.floor(Math.random() * 2)];
							}

							let size = parseFloat(toIncrement.style.getPropertyValue('--size'))
							let data = toIncrement.getElementsByTagName('span')[0]
							data.textContent = parseInt(data.textContent) + 1
							size = size + stepSize;
							if (size <= 1) {
								toIncrement.style.setProperty('--size', size)
							}
						})
					}
					reader.releaseLock();
				}
			})
		})

}
