import React from 'react';
import Box from '@mui/material/Box';
import CircularProgress, {
    circularProgressClasses,
    CircularProgressProps,
}  from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';

function Circular({resultValue, colorValue, progressValue, props}) {
    return (
      <Box sx={{ position: 'relative', display: 'inline-flex'}}>
        <CircularProgress
          variant="determinate"
          sx={(theme) => ({
            color: theme.palette.grey[200],
            ...theme.applyStyles('dark', {
              color: theme.palette.grey[800],
            }),
          })}
          size='7rem'
          thickness={6}
          {...props}
          value={100}
        />
        <CircularProgress
          variant="determinate"
          disableShrink
          sx={() => ({
            color: colorValue,
            position: 'absolute',
            left: 0,
          })}
          size='7rem'
          value={progressValue}
          thickness={6}
          {...props}
        />
        <Box
            sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: 'absolute',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            }}
        >
            <Typography
            variant="h6"
            component="div"
            sx={{ color: colorValue}}
            >
                {resultValue}
            </Typography>
        </Box>
      </Box>
    );
  }


  export default Circular;