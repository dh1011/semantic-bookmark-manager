function toggleEdit(element) {
    const bookmarkItem = element.closest('.bookmark-item');
    bookmarkItem.classList.toggle('editing');
    const summary = bookmarkItem.querySelector('.bookmark-summary');
    const link = bookmarkItem.querySelector('a:first-child');
    if (bookmarkItem.classList.contains('editing')) {
        summary.contentEditable = true;
        summary.focus();
        link.contentEditable = true;
    } else {
        summary.contentEditable = false;
        link.contentEditable = false;
    }
}

function confirmRemove(link) {
    return confirm('Are you sure you want to remove this bookmark?');
}

function saveChanges(element, originalLink) {
    const bookmarkItem = element.closest('.bookmark-item');
    const summary = bookmarkItem.querySelector('.bookmark-summary').innerText;
    const link = bookmarkItem.querySelector('a:first-child').href;
    
    fetch('/update_bookmark', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            original_link: originalLink,
            new_link: link,
            new_summary: summary
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            toggleEdit(element);
        } else {
            alert('Failed to update bookmark');
        }
    });
}