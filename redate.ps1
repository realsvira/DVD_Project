# Redate commits script
# Maps old commit hashes (in chronological order) to new dates

$commits = @(
    @{ Hash = "965834942907cc71a7e5c8abcddb449d35e5b90a"; Date = "2026-10-25T10:17:43+05:30" },
    @{ Hash = "e5fee494f78f6137e72e10cc0ddbf88a2d1cdff3"; Date = "2026-10-27T15:42:11+05:30" },
    @{ Hash = "c84eae2b1209c7231e64a7b090cd44039c5d7c50"; Date = "2026-10-30T11:08:27+05:30" },
    @{ Hash = "21880289797ef96784ff2bcb9d140326bcf9e56a"; Date = "2026-11-02T16:53:04+05:30" },
    @{ Hash = "2366e6ebbc86108c7d494abcffd23791131af954"; Date = "2026-11-05T09:26:38+05:30" },
    @{ Hash = "4f6886ab1e6166430bc4065534a08a00b984f6ce"; Date = "2026-11-08T14:14:52+05:30" },
    @{ Hash = "2ad618171309193ef2749ad11def958dbfbfdf04"; Date = "2026-11-11T10:37:19+05:30" },
    @{ Hash = "6a9411f229c828caea49228c1b22b29fe8e3910f"; Date = "2026-11-14T18:49:33+05:30" }
)

Set-Location "c:\Users\Saumya\DVD_Project"

# Save current branch
$currentBranch = "main"

# Create backup branch
git branch backup-main main 2>&1

# Create orphan branch for rebuilding
git checkout --orphan temp-redate 2>&1
git rm -rf . 2>&1

$first = $true
foreach ($c in $commits) {
    $hash = $c.Hash
    $date = $c.Date

    # Get the commit message
    $msg = git log -1 --format="%B" $hash 2>&1
    # Get the author info
    $authorName = git log -1 --format="%an" $hash 2>&1
    $authorEmail = git log -1 --format="%ae" $hash 2>&1

    # Checkout the tree of this commit
    git read-tree $hash 2>&1
    git checkout $hash -- . 2>&1

    # Set environment variables for dates
    $env:GIT_AUTHOR_DATE = $date
    $env:GIT_COMMITTER_DATE = $date
    $env:GIT_AUTHOR_NAME = $authorName
    $env:GIT_AUTHOR_EMAIL = $authorEmail

    # Commit
    git add -A 2>&1
    $result = git commit --allow-empty -m "$msg" 2>&1
    Write-Host "Committed: $hash -> $date : $result"
}

# Remove env vars
Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_AUTHOR_NAME -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_AUTHOR_EMAIL -ErrorAction SilentlyContinue

# Now replace main with the new branch
git branch -D $currentBranch 2>&1
git branch -m temp-redate $currentBranch 2>&1

Write-Host "`n--- DONE ---"
Write-Host "Verifying new dates:"
git log --oneline --format="%H %ai %s"
