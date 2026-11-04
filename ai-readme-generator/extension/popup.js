
const inspectButton = document.getElementById("inspectButton");
const repoName = document.getElementById("repoName");
const repoLink = document.getElementById("repoLink");
const status = document.getElementById("status");

const results = document.getElementById("results");
const fileCount = document.getElementById("fileCount");
const languageList = document.getElementById("languageList");
const importantFileList = document.getElementById("importantFileList");
const truncatedNotice = document.getElementById("truncatedNotice");

// New README elements
const generateReadmeButton = document.getElementById("generateReadmeButton");
const readmeSection = document.getElementById("readmeSection");
const readmePreview = document.getElementById("readmePreview");
const downloadButton = document.getElementById("downloadButton");

const API_URL = "http://127.0.0.1:8000";

let generatedReadme = "";
let currentRepository = "";

// Analyze the current GitHub repository
inspectButton.addEventListener("click", async () => {
  inspectButton.disabled = true;
  results.hidden = true;
  repoLink.hidden = true;
  status.textContent = "Analyzing repository...";

  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    if (!tab?.url) {
      throw new Error("Could not read the current browser tab.");
    }

    const currentUrl = new URL(tab.url);

    if (
      currentUrl.hostname !== "github.com" ||
      currentUrl.protocol !== "https:"
    ) {
      throw new Error("Please open a GitHub repository page first.");
    }

    const response = await fetch(`${API_URL}/repo-files`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url: tab.url })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Repository analysis failed.");
    }

    currentRepository = data.repository_url;
    repoName.textContent = `${data.owner}/${data.repo}`;
    repoLink.href = data.repository_url;
    repoLink.hidden = false;

    fileCount.textContent = data.total_files ?? 0;

    languageList.replaceChildren();
    importantFileList.replaceChildren();

    const languages = Object.entries(data.languages || {})
      .sort((a, b) => b[1] - a[1]);

    if (languages.length === 0) {
      addListItem(languageList, "No recognized languages found.");
    } else {
      for (const [language, count] of languages) {
        addListItem(languageList, `${language}: ${count} files`);
      }
    }

    const importantFiles = data.important_files || [];

    if (importantFiles.length === 0) {
      addListItem(importantFileList, "No important files detected.");
    } else {
      for (const path of importantFiles.slice(0, 30)) {
        addListItem(importantFileList, path);
      }

      if (importantFiles.length > 30) {
        addListItem(
          importantFileList,
          `And ${importantFiles.length - 30} more files...`
        );
      }
    }

    truncatedNotice.hidden = !data.tree_truncated;
    results.hidden = false;

    // Clear the previous README when a new repository is analyzed.
    generatedReadme = "";
    readmePreview.textContent = "";
    readmeSection.hidden = true;
    downloadButton.disabled = true;

    status.textContent = data.message || "Analysis complete.";
  } catch (error) {
    console.error("Repository analysis error:", error);
    status.textContent = getErrorMessage(error);
  } finally {
    inspectButton.disabled = false;
  }
});

// Generate README using Gemini through your backend

generateReadmeButton.addEventListener("click", async () => {
  generateReadmeButton.disabled = true;
  downloadButton.disabled = true;
  readmeSection.hidden = true;

  try {
    // Step 1: Get the repository URL
    let repositoryUrl = currentRepository;

    if (!repositoryUrl) {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
      });

      if (!tab?.url) {
        throw new Error("Could not read the current browser tab.");
      }

      const url = new URL(tab.url);

      if (
        url.hostname !== "github.com" ||
        url.protocol !== "https:"
      ) {
        throw new Error("Please open a GitHub repository page first.");
      }

      repositoryUrl = tab.url;
    }

    // Step 2: Index the repository automatically
    status.textContent = "Indexing repository...";

    const indexResponse = await fetch(`${API_URL}/index-repo`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url: repositoryUrl })
    });

    const indexData = await indexResponse.json();

    if (!indexResponse.ok) {
      throw new Error(
        indexData.detail || "Repository indexing failed."
      );
    }

    console.log("Repository indexing successful:", indexData);

    // Step 3: Generate the README
    status.textContent = "Generating README with Gemini...";

    const response = await fetch(`${API_URL}/generate-readme`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url: repositoryUrl })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || "README generation failed."
      );
    }

    if (!data.readme || !data.readme.trim()) {
      throw new Error("The backend returned an empty README.");
    }

    // Step 4: Display the README
    generatedReadme = data.readme;
    readmePreview.textContent = generatedReadme;
    readmeSection.hidden = false;
    downloadButton.disabled = false;

    status.textContent = "README generated successfully!";

  } catch (error) {
    console.error("README generation error:", error);
    status.textContent = getErrorMessage(error);

  } finally {
    generateReadmeButton.disabled = false;
  } 
});


// Download the generated README as a Markdown file
downloadButton.addEventListener("click", () => {
  if (!generatedReadme) {
    status.textContent = "Generate a README before downloading.";
    return;
  }

  const blob = new Blob([generatedReadme], {
    type: "text/markdown;charset=utf-8"
  });

  const downloadUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = downloadUrl;
  link.download = "README.md";
  document.body.appendChild(link);
  link.click();
  link.remove();

  URL.revokeObjectURL(downloadUrl);
  status.textContent = "README.md download started.";
});

function addListItem(list, text) {
  const item = document.createElement("li");
  item.textContent = text;
  list.appendChild(item);
}

function getErrorMessage(error) {
  if (error.message === "Failed to fetch") {
    return "Cannot reach the backend. Check that the Python server is running.";
  }

  return error.message || "Something went wrong.";
}
  