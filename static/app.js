const fileInput = document.querySelector('#file');
const upload = document.querySelector('#upload');
const placeholder = document.querySelector('#placeholder');
const preview = document.querySelector('#preview');
const filename = document.querySelector('#filename');
const extractButton = document.querySelector('#extract');
const extractStatus = document.querySelector('#extract-status');
const form = document.querySelector('#card-form');
const registerButton = document.querySelector('#register');
const registerStatus = document.querySelector('#register-status');
const nextButton = document.querySelector('#next');
const fields = [...form.querySelectorAll('input')];

let imageFile = null;
let previewUrl = null;
let busy = false;
let registered = false;

function showStatus(element, message, kind = '', url) {
  element.textContent = message;
  element.dataset.kind = kind;
  if (typeof url !== 'string') return;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' || !/(^|\.)(notion\.so|notion\.site)$/.test(parsed.hostname)) return;
    const link = document.createElement('a');
    link.href = parsed.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Notion で開く';
    element.append(' ', link);
  } catch {
    // A saved entry can still be reported when its link is unavailable.
  }
}

function updateControls() {
  fileInput.disabled = busy || registered;
  extractButton.disabled = busy || registered || !imageFile;
  registerButton.disabled = busy || registered;
  fields.forEach(field => { field.readOnly = busy || registered; });
  nextButton.hidden = !registered;
}

function setImage(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  imageFile = file;
  previewUrl = file ? URL.createObjectURL(file) : null;
  if (previewUrl) preview.src = previewUrl;
  else preview.removeAttribute('src');
  preview.hidden = !file;
  placeholder.hidden = Boolean(file);
  filename.textContent = file ? file.name : 'JPEG・PNG / 20 MB まで';
  updateControls();
}

function selectImage(file) {
  if (busy || registered || !file) return;
  if (!/\.(jpe?g|png)$/i.test(file.name) && !['image/jpeg', 'image/png'].includes(file.type)) {
    showStatus(extractStatus, 'JPEG または PNG を選んでください。', 'error');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showStatus(extractStatus, '20 MB 以下の画像を選んでください。', 'error');
    return;
  }
  if (imageFile && fields.some(field => field.value) &&
      !window.confirm('新しい画像を選ぶと、現在の入力が消えます。続けますか？')) return;
  if (imageFile) form.reset();
  setImage(file);
  showStatus(extractStatus, '「読み取る」を押すと、Mac上で読み取りを始めます。');
  showStatus(registerStatus, '');
}

fileInput.addEventListener('change', () => {
  selectImage(fileInput.files[0]);
  fileInput.value = '';
});

upload.addEventListener('dragover', event => {
  event.preventDefault();
  if (!busy && !registered) upload.classList.add('dragging');
});
upload.addEventListener('dragleave', () => upload.classList.remove('dragging'));
upload.addEventListener('drop', event => {
  event.preventDefault();
  upload.classList.remove('dragging');
  if (!busy && !registered && event.dataTransfer.files.length > 1) {
    showStatus(extractStatus, '名刺画像を1枚ずつ選んでください。', 'error');
    return;
  }
  selectImage(event.dataTransfer.files[0]);
});

async function post(url, body, json = false) {
  const headers = {'X-Card-Request': '1'};
  if (json) headers['Content-Type'] = 'application/json';
  const response = await fetch(url, {method: 'POST', headers, body});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error ?? '処理に失敗しました。');
    error.url = data.url;
    throw error;
  }
  return data;
}

extractButton.addEventListener('click', async () => {
  if (!imageFile || busy || registered) return;
  if (fields.some(field => field.value) &&
      !window.confirm('読み取り結果で現在の入力を置き換えますか？')) return;
  busy = true;
  updateControls();
  showStatus(extractStatus, '読み取り中です。初回はモデルの読み込みに時間がかかります。');
  showStatus(registerStatus, '');
  try {
    const {card} = await post('/api/extract', imageFile);
    fields.forEach(field => { field.value = card[field.name]; });
    showStatus(extractStatus, '読み取りが完了しました。名刺と見比べてください。', 'success');
  } catch (error) {
    showStatus(extractStatus, error.message, 'error');
  } finally {
    busy = false;
    updateControls();
  }
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || registered) return;
  busy = true;
  updateControls();
  showStatus(registerStatus, 'Notion に登録中です。');
  try {
    const body = Object.fromEntries(new FormData(form));
    const result = await post('/api/register', JSON.stringify(body), true);
    registered = true;
    showStatus(registerStatus, 'Notion に登録しました。', 'success', result.url);
  } catch (error) {
    showStatus(registerStatus, error.message, 'error', error.url);
  } finally {
    busy = false;
    updateControls();
  }
});

nextButton.addEventListener('click', () => {
  registered = false;
  form.reset();
  setImage(null);
  showStatus(extractStatus, '');
  showStatus(registerStatus, '');
  fileInput.focus();
});
