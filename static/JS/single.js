// Toggle dropdowns
document.addEventListener('DOMContentLoaded', function () {
    const dropdowns = document.querySelectorAll('.dropdown-wrapper');

    dropdowns.forEach(dropdown => {
        const toggleButton = dropdown.previousElementSibling;
        const menu = dropdown.querySelector('.dropdown-menu');

        toggleButton.addEventListener('click', function (event) {
            event.stopPropagation(); // Prevent click event from bubbling up
            menu.classList.toggle('hidden');
        });

        document.addEventListener('click', function (event) {
            if (!dropdown.contains(event.target)) {
                menu.classList.add('hidden');
            }
        });

        dropdown.addEventListener('mouseover', function () {
            menu.classList.remove('hidden');
        });

        dropdown.addEventListener('mouseleave', function () {
            menu.classList.add('hidden');
        });
    });
});

const fileInput = document.getElementById('singleInputDoc');
const submitButton = document.getElementById('compareSingleDocBtn');

fileInput.addEventListener('change', function () {
    if (fileInput.files.length === 0) {
        submitButton.disabled = true;
        toastr.error('No file selected. Please choose a file to upload.', 'Error', { timeOut: 5000 });
    } else {
        submitButton.disabled = false;
    }
});

submitButton.addEventListener('click', function (event) {
    if (fileInput.files.length === 0) {
        toastr.error('No file selected. Please choose a file to upload.', 'Error', { timeOut: 5000 });
        event.preventDefault(); // Prevent form submission if no file is selected
    }
});

document.getElementById('singleDocumentForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
        toastr.error('No file selected. Please select a file before submitting.', 'Error', { timeOut: 5000 });
        return;
    }

    const formData = new FormData();
    formData.append('document', file);
    document.getElementById('singleResultsContainer').classList.add('hidden');
    toastr.info('Processing your document...', 'Processing', { timeOut: 5000 });

    // Show the loading spinner
    const spinner = document.createElement('div');
    spinner.className = 'spinner mx-auto my-4';
    document.getElementById('singleDocComparison').appendChild(spinner);

    try {
        const response = await fetch('/singleComparison', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log("Received result:", result);  // Debugging line

        if (!response.ok) {
            toastr.error(result.error || 'An unexpected error occurred.', 'Error', { timeOut: 5000 });
        } else {
            displaySingleResults(result);
        }
    } catch (error) {
        toastr.error('Network or server issue. Please try again later.', 'Error', { timeOut: 5000 });
    } finally {
        document.getElementById('singleResultsContainer').classList.remove('hidden'); // Show the results container
        // Remove the loading spinner
        spinner.remove();
    }
});

function displaySingleResults(result) {
    const singleResultsContainer = document.getElementById('singleResults');
    singleResultsContainer.innerHTML = `
        <div class="mt-4 bg-white shadow-md rounded-lg overflow-hidden">
            <div class="bg-blue-500 text-white p-4">
                <h2 class="text-xl font-bold">Results</h2>
            </div>
            <div class="p-4">
                <h3 class="text-lg font-semibold">Similarities with Existing Documents:</h3>
                <div id="similarities-container" class="mb-4"></div>
                <h3 class="text-lg font-semibold mt-4">Overall Similarity:</h3>
                <p id="overall-similarity" class="bg-blue-100 text-blue-800 p-2 rounded"></p>
            </div>
        </div>
    `;

    // Sort results by similarity percentage in descending order
    result.similarities.sort((a, b) => parseFloat(b[1]) - parseFloat(a[1]));

    const similaritiesContainer = document.getElementById('similarities-container');

    result.similarities.forEach(([name, similarity]) => {
        // Remove the file extension
        const nameWithoutExtension = name.replace(/\.[^/.]+$/, '');

        // Determine the background color based on similarity percentage
        let bgColorClass;
        const percentage = parseFloat(similarity);

        if (percentage >= 90) {
            bgColorClass = 'bg-green-200'; // High similarity
        } else if (percentage >= 80) {
            bgColorClass = 'bg-yellow-200'; // Moderate-high similarity
        } else if (percentage >= 70) {
            bgColorClass = 'bg-orange-200'; // Moderate similarity
        } else if (percentage >= 60) {
            bgColorClass = 'bg-red-200'; // Low similarity
        } else {
            bgColorClass = 'bg-gray-200'; // Very low similarity
        }

        const p = document.createElement('p');
        p.className = `mb-2 p-2 rounded ${bgColorClass}`; // Apply background color class
        p.innerHTML = `<strong>${nameWithoutExtension}:</strong> ${similarity.toFixed(2)}%`;
        similaritiesContainer.appendChild(p);
    });

    document.getElementById('overall-similarity').textContent = result.similar_to_existing ?
        'The new document is similar to existing documents.' :
        'The new document is not similar to existing documents.';

    singleResultsContainer.classList.remove('hidden');
    toastr.success('Document processed successfully!', 'Success', { timeOut: 5000 });
}