# TODO #

## 24-10-2024 ##

* Want this to be usable by amateur astronomers, investigate making a web interface.

 - Web front end needs the following features:
    + Upload required files (science observation, standard star observation), back end should handle conversion to FITS if required
    + Choose deconvolution algorithm
    + Set parameters of deconvolution algorithm + display parameters of job.
    + Submit job to queue
    + Report any errors to user, with suggested fixes if possible
    + Report progress to user: state (science obs uploaded, psf obs uploaded, job submitted, job enqueued [queue position], job running [iteration number, max iterations], job cancelled, job completed, job failed)
    + Display result of deconvolution to user
    + Enable download of deconvolved data. Backend should handle conversion to/from FITS if desired.
    + Remember user by using a cookie to run the program
  
  - Backend should handle the following:
    + Convert input file to/from FITS format
    + Limit the number of jobs active at one time
    + Create users when the web-interface asks it to
    + Update associated 'magic number' for a user when the web interface asks us to
    + Delete a user and all their data when the web interface asks us to
    + Delete a user if their 'keep alive' time expires (i.e. the cookie that the user has that holds the 'magic number' is no longer valid)
    + Accept jobs information from web interface, including a field that indicates which user we are running the job for.
    + Communicate each step of process with web front end
    + Respond to events sent by web front end (e.g. cancel job, replace file)
    + Get + save deconv algorithm for job from front end
    + Get + save deconv params for job from front end
    + Run deconvolution code for supplied input files and save output in supplied location, report progress to front end.


Possible architecture:
```
[WEB] --- events ---> [JOB CONTROLLER] -- spawn -->[DECONV JOB]
                         |   |   ^  ^                        |
[WEB] <- progress info ---   |   |  |                        |
                             |   |  --- messages -------------
                   store state   retrieve state
                             |   |
                             v   |
                       [SQLITE DATABASE]

```

All the javascript does is send messages to, and recieve data from, the job controller. All of the state is stored on the controller's end.


Web interface:
```
[] - page
<> - action
{...} - set of things

# NOTE: All of these pages will need to be dynamically generated

# page lists all jobs for user
[JOB LIST] -- has actions --> { <NEW JOB>, <SHOW JOB X>, <DELETE JOB X> }

<NEW JOB> -- displays --> [JOB CREATION]

<SHOW JOB X> -- displays --> [JOB X INFO]

<DELETE JOB X> -- updates --> [JOB LIST]

# Page enables creation of jobs for user
[JOB CREATION] -- has controls -> { <UPLOAD JOB FILES>, <SET JOB PARAMS>, <SUBMIT JOB>, <DISCARD JOB> }
      |
      -- javascript --> Tells server to create a job "X" for current user

<UPLOAD JOB FILES> -- javascript --> uploads images to server for job, replaces them if uploaded again

<SET JOB PARAMS> -- javascript --> sets values for job parameters on web interface

<SUBMIT JOB> -- javascript --> sends job parameters to server -- displays --> [JOB LIST]

<DISCARD JOB> -- javascript --> tells server to remove job "X" and all associated information for current user. -- displays --> [JOB LIST]

# Page shows detailed information on user's job "X"
[JOB X INFO] -- shows information --> { job ID, job status, uploaded files, deconvolved image (if available), error information (if available)}
            |
            --- has controls --> { <DOWNLOAD UPLOADED FILES>, <DOWNLOAD DECONVOLVED IMAGE>, <DISCARD JOB> }

<DOWNLOAD UPLOADED FILES> -- javascript --> downloads the files the user uploaded

<DOWNLOAD DECONVOLVED IMAGE> -- has controls --> {<SET IMAGE FORMAT> } -- javascript --> downloads the deconvolved image in desired format

<SET IMAGE FORMAT> -- javascript --> tells server to convert image to desired format + waits for ready response


```

