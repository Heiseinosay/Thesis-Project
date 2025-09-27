import * as React from 'react';
import { useContext, useState } from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import { useNavigate } from 'react-router-dom';
import { DataContext } from './DataContext';
import '../style/alertstyle.css'

export default function AlertDialog({ forResult }) {

  const [open, setOpen] = React.useState(false);
  const { recordData } = useContext(DataContext);
  const navigate = useNavigate();

  const handleClickOpen = () => {
    if (recordData != null) {
      setOpen(true)
    } else {
      navigate('/record')
    }
  };

  const handleClose = () => {
    setOpen(false);
  }

  const handleRecord = () => {
    setOpen(false);
    navigate('/record');
  };

  const handleResult = () => {
    setOpen(false);
    forResult();
  }

  return (
    <React.Fragment>
      <button className='btn-speaker-identification' variant="outlined" onClick={handleClickOpen}>
        Determine how close the audio is to your voice?
      </button>
      <Dialog
        open={open}
        onClose={handleClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          {"Record Again?"}
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            You already have an existing recording, Would you like to re-record them?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleRecord}>Re-record</Button>
          <Button onClick={handleResult} color='success' autoFocus>
            Use the Existing Recordings
          </Button>
        </DialogActions>
      </Dialog>
    </React.Fragment>
  );
}
